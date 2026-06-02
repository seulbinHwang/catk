#!/usr/bin/env python3

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

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "secrets" / "waymo" / "waymo_storage_state.json"
)
_WAYMO_ORIGIN = "https://waymo.com"
_WAYMO_TERMS_LOCAL_STORAGE_KEY = "datasetChallengeTermsAgreementAccepted"
_SYSTEM_BROWSER_CANDIDATES = {
    "chrome": (
        "/opt/google/chrome/chrome",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
    ),
    "msedge": (
        "microsoft-edge",
        "microsoft-edge-stable",
        "msedge",
    ),
}


def _playwright_install_hint() -> str:
    python_executable = Path(sys.executable).resolve()
    return (
        "Playwright is required.\n"
        f"Current interpreter: {python_executable}\n"
        "Install it with the same interpreter:\n"
        f"  {python_executable} -m pip install playwright==1.52.0\n"
        f"  {python_executable} -m playwright install chromium\n"
        "If `pip install playwright` already succeeded, check that `python -m pip --version` "
        "and `pip --version` point to the same environment."
    )


def _should_use_cdp_attach(args: argparse.Namespace) -> bool:
    return args.browser_name == "chromium" and (
        args.browser_executable_path is not None
        or args.browser_channel in _SYSTEM_BROWSER_CANDIDATES
    )


def _resolve_system_browser_executable(args: argparse.Namespace) -> Path:
    if args.browser_executable_path:
        path = Path(args.browser_executable_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Browser executable does not exist: {path}")
        return path

    if args.browser_channel not in _SYSTEM_BROWSER_CANDIDATES:
        raise ValueError(
            "CDP attach is only supported for system Chromium-based browsers such as "
            "`--browser-channel chrome` or `--browser-channel msedge`."
        )

    for candidate in _SYSTEM_BROWSER_CANDIDATES[args.browser_channel]:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return candidate_path.resolve()
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved).resolve()

    raise FileNotFoundError(
        f"Could not find a browser executable for channel {args.browser_channel!r}. "
        "Install that browser or pass `--browser-executable-path` explicitly."
    )


def _allocate_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _read_log_tail(log_path: Path, max_chars: int = 4000) -> str:
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= max_chars else text[-max_chars:]


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_for_cdp_endpoint(endpoint_url: str, process: subprocess.Popen, log_path: Path) -> str:
    version_url = f"{endpoint_url}/json/version"
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(version_url, timeout=1.0) as response:
                payload = json.load(response)
            websocket_url = payload.get("webSocketDebuggerUrl")
            if websocket_url:
                return str(websocket_url)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
        time.sleep(0.2)

    details = _read_log_tail(log_path)
    raise RuntimeError(
        "Chrome did not expose a DevTools endpoint for Playwright to attach.\n"
        f"Endpoint: {endpoint_url}\n"
        f"Process return code: {process.poll()}\n"
        f"Last connection error: {last_error}\n"
        "If you reused an old `--user-data-dir`, try omitting it to use a fresh temporary profile.\n"
        f"Chrome launch log tail:\n{details}"
    )


def _connect_over_cdp(playwright, websocket_url: str, log_path: Path):
    deadline = time.monotonic() + 10.0
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            return playwright.chromium.connect_over_cdp(websocket_url)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(0.5)

    details = _read_log_tail(log_path)
    raise RuntimeError(
        "Chrome exposed DevTools, but Playwright could not attach over CDP.\n"
        f"WebSocket endpoint: {websocket_url}\n"
        f"Last error: {last_error}\n"
        "If you reused an old `--user-data-dir`, try omitting it to use a fresh temporary profile.\n"
        f"Chrome launch log tail:\n{details}"
    )


def _launch_system_chromium_via_cdp(playwright, args: argparse.Namespace, user_data_dir: Path):
    executable_path = _resolve_system_browser_executable(args)
    port = _allocate_tcp_port()
    endpoint_url = f"http://127.0.0.1:{port}"
    log_path = user_data_dir / "chrome_launch.log"
    launch_cmd = [
        executable_path.as_posix(),
        f"--user-data-dir={user_data_dir}",
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "about:blank",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            launch_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        websocket_url = _wait_for_cdp_endpoint(endpoint_url, process, log_path)
        browser = _connect_over_cdp(playwright, websocket_url, log_path)
    except Exception:
        _terminate_process(process)
        raise

    if not browser.contexts:
        browser.close()
        _terminate_process(process)
        raise RuntimeError(
            "Playwright connected to Chrome over CDP, but no default browser context was exposed."
        )

    return browser, browser.contexts[0], process, executable_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a browser profile, let the user log in to Waymo manually, and save a "
            "Playwright storage_state.json file for later headless submissions."
        )
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT_PATH.as_posix(),
        help=(
            "Where to save storage_state.json. Defaults to the repo-local canonical path "
            f"{_DEFAULT_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--url",
        default="https://waymo.com/open/challenges/2025/sim-agents/",
        help="Challenge page to open before saving storage state.",
    )
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help=(
            "Optional persistent browser profile directory to use while logging in. "
            "If omitted, the script creates a fresh temporary profile for this run and removes "
            "it on exit. If you set it explicitly, use a dedicated directory, not your default "
            "daily Chrome profile."
        ),
    )
    parser.add_argument(
        "--browser-name",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Playwright browser engine to launch.",
    )
    parser.add_argument(
        "--browser-channel",
        default=None,
        help="Optional browser channel such as chrome or msedge.",
    )
    parser.add_argument(
        "--browser-executable-path",
        default=None,
        help="Optional absolute path to a browser executable.",
    )
    return parser.parse_args()


def _prepare_user_data_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.user_data_dir:
        user_data_dir = Path(args.user_data_dir).expanduser().resolve()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        return user_data_dir, False

    temp_dir = Path(tempfile.mkdtemp(prefix="catk-waymo-browser-profile."))
    return temp_dir.resolve(), True


def _wait_for_page_ready(page) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        page.wait_for_timeout(1000)


def _read_waymo_local_storage(page) -> list[dict[str, str]]:
    if not page.url.startswith(_WAYMO_ORIGIN):
        return []

    entries = page.evaluate(
        """() =>
            Object.entries(window.localStorage).map(([name, value]) => ({ name, value }))
        """
    )
    if not isinstance(entries, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        normalized.append({"name": name, "value": value})
    return normalized


def _upsert_waymo_origin_local_storage(
    output_path: Path,
    local_storage_entries: list[dict[str, str]],
) -> None:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    origins = []
    for origin_entry in payload.get("origins", []):
        if not isinstance(origin_entry, dict):
            continue
        if origin_entry.get("origin") == _WAYMO_ORIGIN:
            continue
        origins.append(origin_entry)

    if local_storage_entries:
        origins.append(
            {
                "origin": _WAYMO_ORIGIN,
                "localStorage": local_storage_entries,
            }
        )

    payload["origins"] = origins
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_waymo_submission_page(page) -> tuple[bool, list[dict[str, str]]]:
    body_text = page.locator("body").inner_text(timeout=10000).lower()
    current_url = page.url.lower()

    if "accounts.google.com" in current_url or "/open/auth/login" in current_url:
        raise RuntimeError(
            "The browser is still on the Google / Waymo sign-in page. "
            "Finish sign-in first, then re-run the save script."
        )

    if "sign in to submit" in body_text or "please sign in to continue" in body_text:
        raise RuntimeError(
            "The challenge page still shows the sign-in gate. "
            "Sign in with the target account and make sure the Sim Agents page loads "
            "the upload UI before saving storage_state.json."
        )

    has_review_rules_gate = page.get_by_text("Review rules", exact=False).count() > 0
    if has_review_rules_gate:
        raise RuntimeError(
            "The challenge page still shows `Review rules`. "
            "Open that page on the GUI machine, accept the challenge terms once, wait "
            "until the Sim Agents page shows the upload form, and then save the state again."
        )

    file_input_count = page.locator("input[type='file']").count()
    if file_input_count == 0:
        raise RuntimeError(
            "Could not find a Waymo upload form on the Sim Agents page. "
            "Only save storage_state.json after the page shows the validation / test "
            "submission boxes."
        )

    local_storage_entries = _read_waymo_local_storage(page)
    if not any(
        item.get("name") == _WAYMO_TERMS_LOCAL_STORAGE_KEY and item.get("value") == "true"
        for item in local_storage_entries
    ):
        local_storage_entries = [
            item
            for item in local_storage_entries
            if item.get("name") != _WAYMO_TERMS_LOCAL_STORAGE_KEY
        ]
        local_storage_entries.append(
            {
                "name": _WAYMO_TERMS_LOCAL_STORAGE_KEY,
                "value": "true",
            }
        )

    return True, local_storage_entries


def main() -> None:
    args = _parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(_playwright_install_hint()) from exc

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_data_dir, remove_user_data_dir = _prepare_user_data_dir(args)

    with sync_playwright() as playwright:
        browser = None
        context = None
        external_process = None
        try:
            if _should_use_cdp_attach(args):
                browser, context, external_process, executable_path = _launch_system_chromium_via_cdp(
                    playwright,
                    args,
                    user_data_dir,
                )
                print(
                    "Using external Chromium launch with Playwright CDP attach: "
                    f"{executable_path}"
                )
            else:
                browser_type = getattr(playwright, args.browser_name)
                launch_kwargs = {
                    "headless": False,
                }
                if args.browser_channel:
                    launch_kwargs["channel"] = args.browser_channel
                if args.browser_executable_path:
                    launch_kwargs["executable_path"] = args.browser_executable_path
                if args.browser_name == "chromium":
                    launch_kwargs["chromium_sandbox"] = False

                context = browser_type.launch_persistent_context(
                    user_data_dir=user_data_dir.as_posix(),
                    **launch_kwargs,
                )

            page = context.pages[0] if context.pages else context.new_page()
            page.goto(args.url, wait_until="domcontentloaded")
            _wait_for_page_ready(page)

            profile_mode = "temporary" if remove_user_data_dir else "persistent"
            print(f"Using {profile_mode} browser profile directory: {user_data_dir}")
            if args.browser_name == "chromium" and not args.browser_channel and not args.browser_executable_path:
                print(
                    "Tip: Google sign-in often rejects the bundled Playwright Chromium. "
                    "If that happens, re-run with `--browser-channel chrome` or "
                    "`--browser-executable-path /path/to/chrome`."
                )
            print("1. Sign in to Waymo with the target Google account.")
            print("2. Accept challenge terms if Waymo asks for them.")
            print("3. Confirm the Sim Agents page shows the submission UI instead of the sign-in gate.")
            input("Press Enter here after the page is fully ready to save storage_state.json...")

            page.goto(args.url, wait_until="domcontentloaded")
            _wait_for_page_ready(page)
            _, waymo_local_storage = _validate_waymo_submission_page(page)

            context.storage_state(path=output_path.as_posix())
            _upsert_waymo_origin_local_storage(output_path, waymo_local_storage)
            print(f"Saved Waymo storage state to {output_path}")
            print(
                "Verified that the Sim Agents page showed the upload form and stored the "
                f"`{_WAYMO_TERMS_LOCAL_STORAGE_KEY}` Waymo localStorage flag for headless uploads."
            )
        finally:
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
            elif context is not None:
                try:
                    context.close()
                except Exception:
                    pass
            _terminate_process(external_process)
            if remove_user_data_dir:
                shutil.rmtree(user_data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
