# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from omegaconf import DictConfig, open_dict
import requests

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_DEFAULT_WAYMO_ARCHIVE_NAME = "sim_agents_2025_submission.tar.gz"
_PRIMARY_RANK_ENV_KEYS = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
_SUPPORTED_BROWSER_NAMES = {"chromium", "firefox", "webkit"}
_SUPPORTED_EVALUATION_SETS = {"validation", "test"}
_SIM_AGENTS_CHALLENGE_NAME = "SIM_AGENTS"
_WAYMO_STORAGE_STATE_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "catk_waymo_submission"
_WAYMO_STORAGE_STATE_RUNTIME_FILENAME = "waymo_storage_state.runtime.json"
_WAYMO_STORAGE_STATE_STATUS_FILENAME = "waymo_storage_state.status.json"
_WAYMO_STORAGE_STATE_WAIT_POLL_SECONDS = 1.0
_WAYMO_STORAGE_STATE_WAIT_TIMEOUT_SECONDS = 3600.0
_SYSTEM_CHROMIUM_CANDIDATES = (
    "/opt/google/chrome/chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _playwright_install_hint(prefix: str) -> str:
    python_executable = Path(sys.executable).resolve()
    return (
        f"{prefix}\n"
        f"Current interpreter: {python_executable}\n"
        "Install it with the same interpreter:\n"
        f"  {python_executable} -m pip install -r install/requirements.txt\n"
        f"  {python_executable} -m playwright install chromium\n"
        "If `pip install` already succeeded, check that `python -m pip --version` "
        "and `pip --version` point to the same environment."
    )


@dataclass(frozen=True)
class WaymoSubmissionResult:
    archive_path: Path
    challenge_url: str
    submissions_url: str
    debug_dir: Path


@dataclass(frozen=True)
class _WaymoSubmissionRuntime:
    archive_path: Path
    storage_state_path: Path
    output_dir: Path
    challenge_url: str
    submissions_url: str
    evaluation_set: str
    browser_name: str
    browser_channel: str | None
    browser_executable_path: Path | None
    headless: bool
    chromium_sandbox: bool
    navigation_timeout_ms: int
    upload_timeout_ms: int
    post_submit_wait_ms: int
    poll_submission_status: bool
    poll_timeout_seconds: int
    poll_interval_seconds: int
    save_debug_artifacts: bool
    method_name: str | None

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "waymo_submission_debug"


@dataclass(frozen=True)
class PreparedWaymoStorageState:
    storage_state_path: Path
    cleanup_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _BrowserLaunchCandidate:
    launch_kwargs: dict[str, object]
    description: str


def is_primary_process(env: Mapping[str, str] | None = None) -> bool:
    env = env or os.environ
    for key in _PRIMARY_RANK_ENV_KEYS:
        value = env.get(key)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            log.warning("Ignoring non-integer %s=%r while resolving primary rank.", key, value)
    return True


def resolve_waymo_evaluation_set(action: str, configured_evaluation_set: str | None) -> str:
    expected_by_action = {
        "validate": "validation",
        "test": "test",
    }
    expected = expected_by_action.get(str(action))
    if expected is None:
        raise ValueError(
            "Waymo auto submission only supports action=validate or action=test. "
            f"Received action={action!r}."
        )

    if configured_evaluation_set in (None, ""):
        return expected

    normalized = str(configured_evaluation_set).strip().lower()
    if normalized not in _SUPPORTED_EVALUATION_SETS:
        raise ValueError(
            "waymo_submission.evaluation_set must be one of "
            f"{sorted(_SUPPORTED_EVALUATION_SETS)}, got {configured_evaluation_set!r}."
        )
    if normalized != expected:
        raise ValueError(
            f"action={action!r} expects waymo_submission.evaluation_set={expected!r}, "
            f"got {normalized!r}."
        )
    return normalized


def resolve_waymo_submission_archive_path(
    *,
    output_dir: str | Path,
    archive_path: str | Path | None,
    root_dir: str | Path | None,
) -> Path:
    if archive_path not in (None, ""):
        resolved_override = resolve_configured_path(archive_path, base_dir=root_dir or output_dir)
        if resolved_override is None:
            raise ValueError("archive_path resolution unexpectedly produced None.")
        return resolved_override

    return Path(output_dir).expanduser().resolve() / _DEFAULT_WAYMO_ARCHIVE_NAME


def resolve_configured_path(
    path_value: str | Path | None,
    *,
    base_dir: str | Path | None,
) -> Path | None:
    if path_value in (None, ""):
        return None

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        return path.resolve()

    return (Path(base_dir).expanduser().resolve() / path).resolve()


def maybe_prepare_waymo_storage_state(cfg: DictConfig) -> PreparedWaymoStorageState | None:
    submission_cfg = cfg.get("waymo_submission")
    if not submission_cfg or not bool(submission_cfg.get("enabled")):
        return None

    if not is_waymo_submission_enabled_for_action(
        action=str(cfg.action),
        submit_validate=bool(submission_cfg.get("submit_validate", True)),
        submit_test=bool(submission_cfg.get("submit_test", False)),
    ):
        return None

    root_dir = Path(str(cfg.paths.root_dir)).expanduser().resolve()
    storage_state_path = resolve_configured_path(
        submission_cfg.get("storage_state_path"),
        base_dir=root_dir,
    )
    if storage_state_path is None:
        raise ValueError(
            "waymo_submission.storage_state_path is required when "
            "waymo_submission.enabled=true."
        )

    if storage_state_path.is_file():
        with open_dict(cfg.waymo_submission):
            cfg.waymo_submission.storage_state_path = storage_state_path.as_posix()
        return PreparedWaymoStorageState(storage_state_path=storage_state_path)

    runtime_dir = _resolve_waymo_storage_state_runtime_dir(root_dir=root_dir)
    runtime_storage_state_path = runtime_dir / _WAYMO_STORAGE_STATE_RUNTIME_FILENAME
    status_path = runtime_dir / _WAYMO_STORAGE_STATE_STATUS_FILENAME
    if is_primary_process():
        prepared = _prepare_waymo_storage_state_on_primary(
            configured_path=storage_state_path,
            runtime_dir=runtime_dir,
            runtime_storage_state_path=runtime_storage_state_path,
            status_path=status_path,
        )
    else:
        prepared = _wait_for_waymo_storage_state_from_primary(
            configured_path=storage_state_path,
            status_path=status_path,
        )

    with open_dict(cfg.waymo_submission):
        cfg.waymo_submission.storage_state_path = prepared.storage_state_path.as_posix()

    return prepared


def cleanup_prepared_waymo_storage_state(
    prepared: PreparedWaymoStorageState | None,
) -> None:
    if prepared is None:
        return

    for cleanup_path in prepared.cleanup_paths:
        try:
            if cleanup_path.is_dir():
                cleanup_path.rmdir()
            else:
                cleanup_path.unlink(missing_ok=True)
        except OSError:
            continue


def _prepare_waymo_storage_state_on_primary(
    *,
    configured_path: Path,
    runtime_dir: Path,
    runtime_storage_state_path: Path,
    status_path: Path,
) -> PreparedWaymoStorageState:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(runtime_dir, 0o700)
    except OSError:
        pass

    status_path.unlink(missing_ok=True)
    runtime_storage_state_path.unlink(missing_ok=True)

    try:
        storage_state_text = _prompt_for_waymo_storage_state_json(configured_path)
        _write_private_text_file(runtime_storage_state_path, storage_state_text)
        status_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "storage_state_path": runtime_storage_state_path.as_posix(),
                }
            ),
            encoding="utf-8",
        )
        log.info(
            "Captured Waymo storage state from terminal input for this run only. "
            "The temporary file will be removed on exit."
        )
    except Exception as exc:
        status_path.write_text(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                }
            ),
            encoding="utf-8",
        )
        raise

    return PreparedWaymoStorageState(
        storage_state_path=runtime_storage_state_path,
        cleanup_paths=(runtime_storage_state_path, status_path, runtime_dir),
    )


def _wait_for_waymo_storage_state_from_primary(
    *,
    configured_path: Path,
    status_path: Path,
) -> PreparedWaymoStorageState:
    deadline = time.monotonic() + _WAYMO_STORAGE_STATE_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if status_path.is_file():
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(_WAYMO_STORAGE_STATE_WAIT_POLL_SECONDS)
                continue

            status = str(status_payload.get("status", "")).strip().lower()
            if status == "ready":
                runtime_storage_state_path = Path(
                    str(status_payload["storage_state_path"])
                ).expanduser().resolve()
                return PreparedWaymoStorageState(storage_state_path=runtime_storage_state_path)
            if status == "error":
                raise RuntimeError(str(status_payload.get("message") or ""))

        time.sleep(_WAYMO_STORAGE_STATE_WAIT_POLL_SECONDS)

    raise TimeoutError(
        "Timed out while waiting for rank 0 to receive Waymo storage state JSON from the "
        "terminal. Re-run the command and paste the full contents of "
        f"{configured_path.name} when prompted."
    )


def _resolve_waymo_storage_state_runtime_dir(*, root_dir: Path) -> Path:
    env = os.environ
    key_parts = [
        f"root_dir={root_dir.as_posix()}",
        f"ppid={os.getppid()}",
    ]
    run_id = str(env.get("TORCHELASTIC_RUN_ID", "")).strip()
    if run_id and run_id.lower() != "none":
        key_parts.append(f"run_id={run_id}")
    else:
        for env_key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE"):
            env_value = str(env.get(env_key, "")).strip()
            if env_value:
                key_parts.append(f"{env_key.lower()}={env_value}")
        if len(key_parts) == 1:
            key_parts.append(f"pid={os.getpid()}")

    digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:20]
    return _WAYMO_STORAGE_STATE_RUNTIME_ROOT / digest


def _prompt_for_waymo_storage_state_json(configured_path: Path) -> str:
    input_stream = _resolve_waymo_storage_state_input_stream()
    should_close_stream = input_stream is not sys.stdin
    try:
        print(
            "\n".join(
                [
                    "",
                    "Waymo auto submission could not find the configured storage state file:",
                    f"  {configured_path}",
                    "",
                    "Paste the full contents of your local "
                    "`secrets/waymo/waymo_storage_state.json` now.",
                    "As soon as the pasted text becomes valid JSON, this run will continue.",
                    "Press Ctrl-D to abort.",
                    "",
                ]
            ),
            file=sys.stderr,
            flush=True,
        )
        storage_state = _read_waymo_storage_state_json_from_stream(input_stream)
    finally:
        if should_close_stream:
            input_stream.close()

    return json.dumps(storage_state, ensure_ascii=False, indent=2) + "\n"


def _resolve_waymo_storage_state_input_stream() -> IO[str]:
    if sys.stdin.isatty():
        return sys.stdin

    try:
        tty_stream = open("/dev/tty", "r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Waymo storage state file is missing and there is no interactive terminal "
            "available for pasting it. Either copy the file to the server or launch the "
            "command from an interactive shell."
        ) from exc

    if not tty_stream.isatty():
        tty_stream.close()
        raise RuntimeError(
            "Waymo storage state file is missing and /dev/tty is not interactive."
        )
    return tty_stream


def _read_waymo_storage_state_json_from_stream(input_stream: IO[str]) -> dict:
    collected_lines: list[str] = []
    while True:
        line = input_stream.readline()
        if line == "":
            pasted_text = "".join(collected_lines).strip()
            if not pasted_text:
                raise RuntimeError("No Waymo storage state JSON was provided.")
            raise RuntimeError(
                "Reached end of input before a complete Waymo storage state JSON payload "
                "was received."
            )

        collected_lines.append(line)
        pasted_text = "".join(collected_lines).strip()
        if not pasted_text:
            continue

        try:
            return _parse_waymo_storage_state_json(pasted_text)
        except json.JSONDecodeError:
            continue


def _parse_waymo_storage_state_json(storage_state_text: str) -> dict:
    payload = json.loads(storage_state_text)
    if not isinstance(payload, dict):
        raise ValueError("Waymo storage state must be a JSON object.")

    cookies = payload.get("cookies")
    if not isinstance(cookies, list):
        raise ValueError("Waymo storage state JSON must contain a `cookies` list.")

    origins = payload.get("origins")
    if origins is not None and not isinstance(origins, list):
        raise ValueError(
            "Waymo storage state JSON must contain an `origins` list when it is present."
        )

    return payload


def _write_private_text_file(path: Path, text: str) -> None:
    file_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
        output_file.write(text)


def maybe_submit_waymo_submission(cfg: DictConfig) -> WaymoSubmissionResult | None:
    submission_cfg = cfg.get("waymo_submission")
    if not submission_cfg or not bool(submission_cfg.get("enabled")):
        return None

    if not is_waymo_submission_enabled_for_action(
        action=str(cfg.action),
        submit_validate=bool(submission_cfg.get("submit_validate", True)),
        submit_test=bool(submission_cfg.get("submit_test", False)),
    ):
        if str(cfg.action) in {"validate", "test"}:
            log.info(
                "Waymo auto submission is disabled for action=%s by config. "
                "Set waymo_submission.submit_%s=true to arm it.",
                cfg.action,
                str(cfg.action),
            )
        else:
            log.info("Skipping Waymo auto submission for unsupported action=%s.", cfg.action)
        return None

    if not is_primary_process():
        log.info("Skipping Waymo auto submission on a non-primary process.")
        return None

    runtime = _build_runtime_config(cfg)

    log.info(
        "Submitting %s to Waymo %s set via %s.",
        runtime.archive_path,
        runtime.evaluation_set,
        runtime.challenge_url,
    )
    uploader = _WaymoSubmissionUploader(runtime=runtime)
    return uploader.submit()


def is_waymo_submission_enabled_for_action(
    *,
    action: str,
    submit_validate: bool,
    submit_test: bool,
) -> bool:
    normalized_action = str(action).strip().lower()
    if normalized_action == "validate":
        return submit_validate
    if normalized_action == "test":
        return submit_test
    return False


def _build_runtime_config(cfg: DictConfig) -> _WaymoSubmissionRuntime:
    submission_cfg = cfg.waymo_submission
    output_dir = Path(str(cfg.paths.output_dir)).expanduser().resolve()
    root_dir = Path(str(cfg.paths.root_dir)).expanduser().resolve()
    evaluation_set = resolve_waymo_evaluation_set(
        action=str(cfg.action),
        configured_evaluation_set=submission_cfg.get("evaluation_set"),
    )
    archive_path = resolve_waymo_submission_archive_path(
        output_dir=output_dir,
        archive_path=submission_cfg.get("archive_path"),
        root_dir=root_dir,
    )
    storage_state_path = resolve_configured_path(
        submission_cfg.get("storage_state_path"),
        base_dir=root_dir,
    )
    if storage_state_path is None:
        raise ValueError(
            "waymo_submission.storage_state_path is required when "
            "waymo_submission.enabled=true."
        )
    if not storage_state_path.is_file():
        raise FileNotFoundError(
            "Waymo storage state file was not found: "
            f"{storage_state_path}. Create it on a GUI machine and copy it to the server."
        )
    if not archive_path.is_file():
        raise FileNotFoundError(
            "Waymo submission archive was not found: "
            f"{archive_path}. The validation/test run must finish and produce "
            f"{_DEFAULT_WAYMO_ARCHIVE_NAME} before auto submission can start."
        )

    browser_name = str(submission_cfg.get("browser_name", "chromium")).strip().lower()
    if browser_name not in _SUPPORTED_BROWSER_NAMES:
        raise ValueError(
            f"waymo_submission.browser_name must be one of {sorted(_SUPPORTED_BROWSER_NAMES)}, "
            f"got {browser_name!r}."
        )

    browser_executable_path = resolve_configured_path(
        submission_cfg.get("browser_executable_path"),
        base_dir=root_dir,
    )
    method_name = _resolve_method_name(cfg)

    return _WaymoSubmissionRuntime(
        archive_path=archive_path,
        storage_state_path=storage_state_path,
        output_dir=output_dir,
        challenge_url=str(submission_cfg.get("challenge_url")),
        submissions_url=str(submission_cfg.get("submissions_url")),
        evaluation_set=evaluation_set,
        browser_name=browser_name,
        browser_channel=submission_cfg.get("browser_channel"),
        browser_executable_path=browser_executable_path,
        headless=bool(submission_cfg.get("headless")),
        chromium_sandbox=bool(submission_cfg.get("chromium_sandbox")),
        navigation_timeout_ms=int(submission_cfg.get("navigation_timeout_ms")),
        upload_timeout_ms=int(submission_cfg.get("upload_timeout_ms")),
        post_submit_wait_ms=int(submission_cfg.get("post_submit_wait_ms")),
        poll_submission_status=bool(submission_cfg.get("poll_submission_status")),
        poll_timeout_seconds=int(submission_cfg.get("poll_timeout_seconds")),
        poll_interval_seconds=int(submission_cfg.get("poll_interval_seconds")),
        save_debug_artifacts=bool(submission_cfg.get("save_debug_artifacts")),
        method_name=method_name,
    )


def _resolve_method_name(cfg: DictConfig) -> str | None:
    model_cfg = cfg.get("model")
    if not model_cfg:
        return None

    model_config = model_cfg.get("model_config")
    if not model_config:
        return None

    submission_cfg = model_config.get("sim_agents_submission")
    if not submission_cfg:
        return None

    method_name = submission_cfg.get("method_name")
    if method_name in (None, ""):
        return None
    return str(method_name)


class _WaymoSubmissionUploader:
    def __init__(self, runtime: _WaymoSubmissionRuntime) -> None:
        self.runtime = runtime

    def submit(self) -> WaymoSubmissionResult:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            log.warning(
                "%s Falling back to direct HTTP upload using the saved Waymo session.",
                _playwright_install_hint("Playwright is required for browser-based Waymo auto submission."),
            )
            return self._submit_via_http()

        with sync_playwright() as playwright:
            try:
                browser = self._launch_browser(playwright)
            except Exception as exc:
                log.warning(
                    "Failed to launch Playwright %s browser (%s). "
                    "Falling back to direct HTTP upload using the saved Waymo session.",
                    self.runtime.browser_name,
                    exc,
                )
                return self._submit_via_http()
            context = None
            page = None
            try:
                context = browser.new_context(
                    storage_state=self.runtime.storage_state_path.as_posix(),
                    accept_downloads=False,
                )
                page = context.new_page()
                page.set_default_timeout(self.runtime.navigation_timeout_ms)

                self._goto(page, self.runtime.challenge_url)
                self._ensure_signed_in(page)

                submit_section = self._find_submission_section(page)
                if submit_section is None:
                    self._save_debug_artifacts(page, "missing-submit-section")
                    raise RuntimeError(
                        "Could not find the Waymo submission form on the challenge page. "
                        "Open the saved debug HTML/PNG and verify that the account can see "
                        f"the {self.runtime.evaluation_set} upload section."
                    )

                if self._submission_section_requires_terms_review(submit_section):
                    self._save_debug_artifacts(page, "terms-gate")
                    raise RuntimeError(
                        "Waymo still shows the `Review rules` gate instead of the upload form. "
                        "On a GUI machine, open the Sim Agents challenge page, accept the "
                        "challenge rules until the file upload boxes are visible, then refresh "
                        "waymo_storage_state.json and copy it to the server."
                    )

                file_input = self._find_file_input(page, submit_section)
                if file_input is None:
                    self._save_debug_artifacts(page, "missing-file-input")
                    raise RuntimeError(
                        "Could not find a file input in the Waymo submission form."
                    )

                file_input.set_input_files(self.runtime.archive_path.as_posix())
                submitted = self._submit_form(page, submit_section, file_input)
                if not submitted:
                    log.warning(
                        "No explicit submit trigger was found after attaching %s. "
                        "If Waymo changed the DOM, inspect %s for the current structure.",
                        self.runtime.archive_path.name,
                        self.runtime.debug_dir,
                    )

                page.wait_for_timeout(self.runtime.post_submit_wait_ms)
                self._save_debug_artifacts(
                    page,
                    f"challenge-page-after-{self.runtime.evaluation_set}-submit",
                )

                self._goto(page, self.runtime.submissions_url)
                self._ensure_signed_in(page)
                self._save_debug_artifacts(page, "submissions-page")

                if (
                    self.runtime.poll_submission_status
                    and self.runtime.poll_timeout_seconds > 0
                ):
                    self._poll_submissions_page(page)

                context.storage_state(path=self.runtime.storage_state_path.as_posix())
            except PlaywrightTimeoutError as exc:
                if page is not None:
                    self._save_debug_artifacts(page, "timeout")
                raise RuntimeError(
                    "Timed out while interacting with the Waymo submission UI. "
                    f"Inspect {self.runtime.debug_dir} for the captured HTML/PNG."
                ) from exc
            except Exception:
                if page is not None:
                    self._save_debug_artifacts(page, "failed")
                raise
            finally:
                if context is not None:
                    context.close()
                browser.close()

        return WaymoSubmissionResult(
            archive_path=self.runtime.archive_path,
            challenge_url=self.runtime.challenge_url,
            submissions_url=self.runtime.submissions_url,
            debug_dir=self.runtime.debug_dir,
        )

    def _submit_via_http(self) -> WaymoSubmissionResult:
        session = self._build_http_session()
        challenge_response = session.get(
            self.runtime.challenge_url,
            timeout=self._requests_timeout_seconds(),
        )
        self._ensure_signed_in_http(challenge_response)
        self._save_http_debug_artifact(
            f"http-challenge-page-before-{self.runtime.evaluation_set}-submit",
            challenge_response.text,
        )

        archive_size = self.runtime.archive_path.stat().st_size
        upload_metadata_response = session.get(
            "https://waymo.com/open/api/createUploadUrl.json",
            params={
                "challenge": _SIM_AGENTS_CHALLENGE_NAME,
                "submissionType": self.runtime.evaluation_set.upper(),
                "filename": self.runtime.archive_path.name,
                "contentLength": str(archive_size),
            },
            headers={"Referer": self.runtime.challenge_url},
            timeout=self._requests_timeout_seconds(),
        )
        self._raise_for_status(upload_metadata_response, "prepare the Waymo upload")
        upload_metadata = upload_metadata_response.json()
        if not bool(upload_metadata.get("success")):
            raise RuntimeError(
                "Waymo did not accept the upload preparation request: "
                f"{upload_metadata!r}"
            )

        upload_url = upload_metadata.get("uploadUrl")
        if isinstance(upload_url, list):
            upload_url = upload_url[0] if upload_url else None
        if not upload_url:
            raise RuntimeError(
                "Waymo upload preparation response did not include an upload URL."
            )

        content_type = upload_metadata.get("contentType")
        if not content_type:
            raise RuntimeError(
                "Waymo upload preparation response did not include a content type."
            )

        with self.runtime.archive_path.open("rb") as archive_file:
            upload_response = requests.put(
                str(upload_url),
                data=archive_file,
                headers={
                    "Content-Type": str(content_type),
                    "Content-Length": str(archive_size),
                },
                timeout=(self._requests_timeout_seconds(), None),
            )
        self._raise_for_status(upload_response, "upload the Waymo submission archive")

        submissions_response = session.get(
            self.runtime.submissions_url,
            timeout=self._requests_timeout_seconds(),
        )
        self._ensure_signed_in_http(submissions_response)
        self._save_http_debug_artifact("http-submissions-page", submissions_response.text)

        if self.runtime.poll_submission_status and self.runtime.poll_timeout_seconds > 0:
            self._poll_submissions_page_http(session)

        return WaymoSubmissionResult(
            archive_path=self.runtime.archive_path,
            challenge_url=self.runtime.challenge_url,
            submissions_url=self.runtime.submissions_url,
            debug_dir=self.runtime.debug_dir,
        )

    def _launch_browser(self, playwright):
        browser_type = getattr(playwright, self.runtime.browser_name, None)
        if browser_type is None:
            raise ValueError(
                f"Unsupported Playwright browser: {self.runtime.browser_name!r}."
            )
        launch_errors: list[str] = []
        for candidate in self._build_browser_launch_candidates():
            try:
                return browser_type.launch(**candidate.launch_kwargs)
            except Exception as exc:
                launch_errors.append(f"{candidate.description}: {exc}")

        if launch_errors:
            raise RuntimeError(
                "All browser launch attempts failed:\n- " + "\n- ".join(launch_errors)
            )
        raise RuntimeError("No browser launch candidates were available.")

    def _build_browser_launch_candidates(self) -> list[_BrowserLaunchCandidate]:
        launch_kwargs: dict[str, object] = {
            "headless": self.runtime.headless,
        }
        launch_env = self._build_browser_launch_env()
        if launch_env is not None:
            launch_kwargs["env"] = launch_env

        if self.runtime.browser_name == "chromium":
            launch_kwargs["chromium_sandbox"] = self.runtime.chromium_sandbox

        candidates: list[_BrowserLaunchCandidate] = []

        if self.runtime.browser_channel:
            candidates.append(
                _BrowserLaunchCandidate(
                    launch_kwargs={**launch_kwargs, "channel": str(self.runtime.browser_channel)},
                    description=f"Playwright channel={self.runtime.browser_channel}",
                )
            )

        if self.runtime.browser_executable_path is not None:
            candidates.append(
                _BrowserLaunchCandidate(
                    launch_kwargs={
                        **launch_kwargs,
                        "executable_path": self.runtime.browser_executable_path.as_posix(),
                    },
                    description=(
                        "configured executable_path="
                        f"{self.runtime.browser_executable_path.as_posix()}"
                    ),
                )
            )

        if not candidates:
            candidates.append(
                _BrowserLaunchCandidate(
                    launch_kwargs=dict(launch_kwargs),
                    description=f"Playwright bundled {self.runtime.browser_name}",
                )
            )

        if self.runtime.browser_name == "chromium":
            for executable_path in self._detect_chromium_executable_candidates():
                executable_path_str = executable_path.as_posix()
                if any(
                    candidate.launch_kwargs.get("executable_path") == executable_path_str
                    for candidate in candidates
                ):
                    continue
                candidates.append(
                    _BrowserLaunchCandidate(
                        launch_kwargs={**launch_kwargs, "executable_path": executable_path_str},
                        description=f"detected chromium executable={executable_path_str}",
                    )
                )

        return candidates

    def _build_browser_launch_env(self) -> dict[str, str] | None:
        env = dict(os.environ)
        conda_prefix = str(env.get("CONDA_PREFIX", "")).strip()
        if not conda_prefix:
            return None

        conda_lib = Path(conda_prefix).expanduser().resolve() / "lib"
        if not conda_lib.is_dir():
            return None

        existing_ld_library_path = str(env.get("LD_LIBRARY_PATH", "")).strip()
        ld_library_entries = [
            entry for entry in existing_ld_library_path.split(":") if entry
        ]
        conda_lib_str = conda_lib.as_posix()
        if conda_lib_str not in ld_library_entries:
            ld_library_entries.insert(0, conda_lib_str)
        env["LD_LIBRARY_PATH"] = ":".join(ld_library_entries)
        return env

    def _detect_chromium_executable_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        seen: set[str] = set()

        env_candidates = (
            os.environ.get("WAYMO_SUBMISSION_BROWSER"),
            os.environ.get("CHROME_BIN"),
            os.environ.get("GOOGLE_CHROME_BIN"),
        )
        for candidate in env_candidates:
            self._append_executable_candidate(candidates, seen, candidate)

        for candidate in _SYSTEM_CHROMIUM_CANDIDATES:
            self._append_executable_candidate(candidates, seen, candidate)

        for binary_name in (
            "google-chrome",
            "google-chrome-stable",
            "chrome",
            "chromium",
            "chromium-browser",
        ):
            resolved = shutil.which(binary_name)
            self._append_executable_candidate(candidates, seen, resolved)

        playwright_cache_root = Path.home() / ".cache" / "ms-playwright"
        if playwright_cache_root.is_dir():
            for binary_path in sorted(playwright_cache_root.glob("chromium-*/chrome-linux/chrome")):
                self._append_executable_candidate(candidates, seen, binary_path)

        return candidates

    @staticmethod
    def _append_executable_candidate(
        candidates: list[Path],
        seen: set[str],
        candidate: str | Path | None,
    ) -> None:
        if not candidate:
            return

        path = Path(str(candidate)).expanduser()
        if not path.is_file():
            return

        resolved = path.resolve()
        resolved_str = resolved.as_posix()
        if resolved_str in seen:
            return

        seen.add(resolved_str)
        candidates.append(resolved)

    def _goto(self, page, url: str) -> None:
        page.goto(url, wait_until="domcontentloaded", timeout=self.runtime.navigation_timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=self.runtime.navigation_timeout_ms)
        except Exception:
            page.wait_for_timeout(1000)

    def _ensure_signed_in(self, page) -> None:
        current_url = page.url.lower()
        if "accounts.google.com" in current_url or "/open/auth/login" in current_url:
            self._save_debug_artifacts(page, "auth-required")
            raise RuntimeError(
                "Waymo redirected to Google sign-in. The storage_state.json file is missing, "
                "expired, or does not have permission to access the challenge submission UI."
            )

        auth_gate_text = re.compile(r"sign in to submit|please sign in to continue", re.IGNORECASE)
        if self._locator_count(page.get_by_text(auth_gate_text)) > 0:
            self._save_debug_artifacts(page, "auth-gate")
            raise RuntimeError(
                "The Waymo challenge page still shows the sign-in gate. "
                "Refresh the saved storage_state.json from a logged-in browser session."
            )

    def _find_submission_section(self, page):
        section_heading = re.compile(
            rf"submit to {re.escape(self.runtime.evaluation_set)} set",
            re.IGNORECASE,
        )
        heading = self._first_visible(page.get_by_text(section_heading))
        if heading is not None:
            # Prefer the smallest ancestor that actually owns the target file input / submit
            # controls. The broader "Submit" section contains both test and validation forms.
            targeted_ancestors = (
                heading.locator("xpath=ancestor::*[.//input[@type='file']][1]"),
                heading.locator(
                    "xpath=ancestor::*[.//button[@type='submit'] or .//input[@type='submit']][1]"
                ),
                heading.locator("xpath=ancestor::div[1]"),
                heading.locator("xpath=ancestor::form[1]"),
                heading.locator("xpath=ancestor::section[1]"),
            )
            for ancestor in targeted_ancestors:
                if self._locator_count(ancestor) > 0:
                    return ancestor.first

        section_candidates = [
            page.locator("section").filter(has_text=section_heading),
            page.locator("main").locator("section").filter(has_text=section_heading),
            page.locator("form").filter(has_text=section_heading),
        ]
        for candidate in section_candidates:
            section = self._first_visible(candidate)
            if section is not None:
                return section
        return None

    def _find_file_input(self, page, submit_section):
        locator = submit_section.locator("input[type='file']")
        if self._locator_count(locator) > 0:
            return locator.first
        return None

    def _submission_section_requires_terms_review(self, submit_section) -> bool:
        if self._locator_count(submit_section.locator("[data-terms-gate]")) > 0:
            return True
        review_rules = submit_section.get_by_text(re.compile(r"review rules", re.IGNORECASE))
        return self._locator_count(review_rules) > 0

    def _submit_form(self, page, submit_section, file_input) -> bool:
        page.wait_for_timeout(1000)

        button_patterns = (
            re.compile(r"submit", re.IGNORECASE),
            re.compile(r"upload", re.IGNORECASE),
            re.compile(r"publish", re.IGNORECASE),
            re.compile(r"send", re.IGNORECASE),
        )
        for pattern in button_patterns:
            button = self._first_visible(submit_section.get_by_role("button", name=pattern))
            if button is None:
                continue
            button.click(timeout=self.runtime.upload_timeout_ms)
            self._wait_after_submit(page)
            return True

        submit_input = self._first_visible(
            submit_section.locator("button[type='submit'], input[type='submit']")
        )
        if submit_input is not None:
            submit_input.click(timeout=self.runtime.upload_timeout_ms)
            self._wait_after_submit(page)
            return True

        form_locator = file_input.locator("xpath=ancestor::form[1]")
        if self._locator_count(form_locator) > 0:
            form_handle = form_locator.first.element_handle()
            if form_handle is not None:
                page.evaluate(
                    "(form) => form.requestSubmit ? form.requestSubmit() : form.submit()",
                    form_handle,
                )
                self._wait_after_submit(page)
                return True

        return False

    def _wait_after_submit(self, page) -> None:
        try:
            page.wait_for_load_state("networkidle", timeout=self.runtime.upload_timeout_ms)
        except Exception:
            page.wait_for_timeout(self.runtime.post_submit_wait_ms)

    def _poll_submissions_page(self, page) -> None:
        match_terms = [self.runtime.archive_path.name]
        if self.runtime.method_name:
            match_terms.append(self.runtime.method_name)

        deadline = time.monotonic() + self.runtime.poll_timeout_seconds
        while time.monotonic() < deadline:
            body_text = page.locator("body").inner_text()
            matched_term = next((term for term in match_terms if term in body_text), None)
            if matched_term is not None:
                text_window = _extract_text_window(body_text, matched_term)
                log.info("Waymo submissions page match: %s", text_window)
                lowered = text_window.lower()
                if any(
                    token in lowered
                    for token in (
                        "score",
                        "failed",
                        "error",
                        "complete",
                        "completed",
                        "success",
                    )
                ):
                    return

            page.wait_for_timeout(self.runtime.poll_interval_seconds * 1000)
            page.reload(
                wait_until="domcontentloaded",
                timeout=self.runtime.navigation_timeout_ms,
            )

        log.warning(
            "Timed out while polling %s for the new Waymo submission status.",
            self.runtime.submissions_url,
        )

    def _poll_submissions_page_http(self, session: requests.Session) -> None:
        match_terms = [_SIM_AGENTS_CHALLENGE_NAME]
        if self.runtime.method_name:
            match_terms.append(self.runtime.method_name)

        deadline = time.monotonic() + self.runtime.poll_timeout_seconds
        while time.monotonic() < deadline:
            response = session.get(
                self.runtime.submissions_url,
                timeout=self._requests_timeout_seconds(),
            )
            self._ensure_signed_in_http(response)
            body_text = response.text

            matched_term = next((term for term in match_terms if term in body_text), None)
            if matched_term is not None:
                text_window = _extract_text_window(body_text, matched_term)
                log.info("Waymo submissions page match: %s", text_window)
                lowered = text_window.lower()
                if any(
                    token in lowered
                    for token in (
                        "score",
                        "failed",
                        "error",
                        "complete",
                        "completed",
                        "success",
                    )
                ):
                    return

            time.sleep(self.runtime.poll_interval_seconds)

        log.warning(
            "Timed out while polling %s for the new Waymo submission status.",
            self.runtime.submissions_url,
        )

    def _save_debug_artifacts(self, page, prefix: str) -> None:
        if not self.runtime.save_debug_artifacts:
            return

        try:
            self.runtime.debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = self.runtime.debug_dir / f"{prefix}-{timestamp}.html"
            png_path = self.runtime.debug_dir / f"{prefix}-{timestamp}.png"
            html_path.write_text(page.content(), encoding="utf-8")
            page.screenshot(path=png_path.as_posix(), full_page=True)
        except Exception as exc:
            log.warning("Failed to persist Waymo debug artifacts: %s", exc)

    def _save_http_debug_artifact(self, prefix: str, html: str) -> None:
        if not self.runtime.save_debug_artifacts:
            return

        try:
            self.runtime.debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            html_path = self.runtime.debug_dir / f"{prefix}-{timestamp}.html"
            html_path.write_text(html, encoding="utf-8")
        except Exception as exc:
            log.warning("Failed to persist Waymo HTTP debug artifacts: %s", exc)

    def _build_http_session(self) -> requests.Session:
        state = json.loads(self.runtime.storage_state_path.read_text(encoding="utf-8"))
        session = requests.Session()
        for cookie in state.get("cookies", []):
            session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )
        return session

    def _ensure_signed_in_http(self, response: requests.Response) -> None:
        self._raise_for_status(response, "open the Waymo challenge page")
        current_url = response.url.lower()
        if "accounts.google.com" in current_url or "/open/auth/login" in current_url:
            self._save_http_debug_artifact("http-auth-required", response.text)
            raise RuntimeError(
                "Waymo redirected to Google sign-in. The storage_state.json file is missing, "
                "expired, or does not have permission to access the challenge submission UI."
            )

        auth_gate_text = re.compile(r"sign in to submit|please sign in to continue", re.IGNORECASE)
        if auth_gate_text.search(response.text):
            self._save_http_debug_artifact("http-auth-gate", response.text)
            raise RuntimeError(
                "The Waymo challenge page still shows the sign-in gate. "
                "Refresh the saved storage_state.json from a logged-in browser session."
            )

    @staticmethod
    def _raise_for_status(response: requests.Response, action: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Failed to {action}. HTTP {response.status_code}: {response.text[:1000]}"
            ) from exc

    def _requests_timeout_seconds(self) -> float:
        return max(self.runtime.navigation_timeout_ms, self.runtime.upload_timeout_ms) / 1000.0

    @staticmethod
    def _locator_count(locator) -> int:
        try:
            return int(locator.count())
        except Exception:
            return 0

    def _first_visible(self, locator):
        locator_count = self._locator_count(locator)
        for idx in range(locator_count):
            candidate = locator.nth(idx)
            try:
                if candidate.is_visible():
                    return candidate
            except Exception:
                continue
        if locator_count > 0:
            return locator.first
        return None


def _extract_text_window(text: str, needle: str, radius: int = 240) -> str:
    match_index = text.find(needle)
    if match_index < 0:
        return ""

    start = max(0, match_index - radius)
    end = min(len(text), match_index + len(needle) + radius)
    return " ".join(text[start:end].split())
