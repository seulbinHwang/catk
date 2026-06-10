#!/usr/bin/env python3
"""Sequentially launch the fm-sf-6 one-epoch self-forced DMD beta sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BASE_WRAPPER = Path(__file__).with_name(
    "launch_self_forced_dmd_h100x8_fmsf6_sf_anchor_warm0_beta08_ep1_static_pod.py"
)
DEFAULT_BETAS = ("0.8", "0.9", "0.95", "0.99", "0.995")
DEFAULT_NAMESPACE = "p-sp-labs-reai-training"
DEFAULT_POD = "fm-sf-6"
DEFAULT_CONTAINER = "main"
DEFAULT_LOG_ROOT = "/mnt/nuplan/projects/catk/logs/tmux_static_multinode"


def beta_tag(beta: str) -> str:
    return "beta" + beta.replace(".", "")


def task_name_for_beta(beta: str) -> str:
    tag = beta_tag(beta)
    return (
        f"flow_self_forced_dmd_h100x8_fmsf6_sfanchor_stride1_{tag}_"
        "epoch061_x5f9g0ce_activecontrol_sample16_backprop8_lr5e-5_"
        "bs5to2_frac025_ep1_warm0_middle_val1_agent_oomretry"
    )


def session_for_beta(beta: str) -> str:
    return (
        "catk-self-forced-dmd-h100x8-fmsf6-sfanchor-stride1-"
        f"warm0-{beta_tag(beta)}-ep1-lr5e5"
    )


def kubectl_exec(
    *,
    namespace: str,
    pod: str,
    container: str,
    script: str,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod,
            "-c",
            container,
            "--",
            "bash",
            "-lc",
            script,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def remote_log_path(task_name: str, pod: str) -> str:
    return f"{DEFAULT_LOG_ROOT}/{task_name}/{pod}.tmux.log"


def remote_success_status(
    *,
    namespace: str,
    pod: str,
    container: str,
    log_path: str,
) -> str:
    probe = f"""
set +e
if [ ! -f {log_path!r} ]; then
  echo MISSING
  exit 0
fi
if grep -q 'training completed successfully' {log_path!r} && grep -q 'run.py DONE!!!' {log_path!r}; then
  echo SUCCESS
  exit 0
fi
if grep -Eq 'non-OOM failure|reached MIN_BS|refusing to start|checkpoint plan sync failed|timed out waiting' {log_path!r}; then
  echo FAILURE
  tail -80 {log_path!r}
  exit 0
fi
echo RUNNING
tail -20 {log_path!r}
"""
    result = kubectl_exec(
        namespace=namespace,
        pod=pod,
        container=container,
        script=probe,
        capture=True,
    )
    return result.stdout or ""


def wait_for_success(
    *,
    beta: str,
    task_name: str,
    namespace: str,
    pod: str,
    container: str,
    poll_interval_sec: int,
    timeout_hours: float,
) -> None:
    deadline = time.monotonic() + timeout_hours * 3600.0
    log_path = remote_log_path(task_name, pod)
    while time.monotonic() < deadline:
        status_text = remote_success_status(
            namespace=namespace,
            pod=pod,
            container=container,
            log_path=log_path,
        )
        first_line = status_text.splitlines()[0] if status_text.splitlines() else "UNKNOWN"
        print(f"[beta-sweep] beta={beta} status={first_line} log={log_path}", flush=True)
        if first_line == "SUCCESS":
            return
        if first_line == "FAILURE":
            raise RuntimeError(f"beta={beta} failed; tail follows:\n{status_text}")
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Timed out waiting for beta={beta} after {timeout_hours} hours.")


def launch_beta(
    *,
    beta: str,
    replace: bool,
    passthrough_args: list[str],
) -> None:
    task_name = task_name_for_beta(beta)
    session = session_for_beta(beta)
    command = [
        sys.executable,
        str(BASE_WRAPPER),
        "--task-name",
        task_name,
        "--session",
        session,
        "--extra-hydra-overrides",
        f"model.model_config.self_forced.beta={beta}",
        *passthrough_args,
    ]
    if replace:
        command.append("--replace")
    print("[beta-sweep] launching:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stop_beta(*, beta: str, passthrough_args: list[str]) -> None:
    command = [
        sys.executable,
        str(BASE_WRAPPER),
        "--task-name",
        task_name_for_beta(beta),
        "--session",
        session_for_beta(beta),
        "--stop",
        *passthrough_args,
    ]
    subprocess.run(command, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run beta=0.8/0.9/0.95/0.99/0.995 self-forced DMD experiments "
            "sequentially on fm-sf-6."
        )
    )
    parser.add_argument("--betas", nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--poll-interval-sec", type=int, default=300)
    parser.add_argument("--timeout-hours-per-run", type=float, default=24.0)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--keep-session", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.poll_interval_sec < 1:
        parser.error("--poll-interval-sec must be >= 1")
    if args.timeout_hours_per_run <= 0:
        parser.error("--timeout-hours-per-run must be > 0")
    return args


def main() -> int:
    args = parse_args()
    passthrough = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    for beta in args.betas:
        task_name = task_name_for_beta(beta)
        print(f"[beta-sweep] beta={beta} task={task_name}", flush=True)
        if args.dry_run:
            launch_beta(beta=beta, replace=args.replace, passthrough_args=[*passthrough, "--dry-run"])
            continue
        launch_beta(beta=beta, replace=args.replace, passthrough_args=passthrough)
        wait_for_success(
            beta=beta,
            task_name=task_name,
            namespace=args.namespace,
            pod=args.pod,
            container=args.container,
            poll_interval_sec=args.poll_interval_sec,
            timeout_hours=args.timeout_hours_per_run,
        )
        if not args.keep_session:
            stop_beta(beta=beta, passthrough_args=passthrough)
    print("[beta-sweep] all beta runs completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
