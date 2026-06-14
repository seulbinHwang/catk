#!/usr/bin/env python3
"""Launch self_forcing_w_road RoaD flow fine-tuning on the static testas A100x7 pod.

The launcher never creates, deletes, or restarts pods. It prepares a clean
self_forcing_w_road checkout under /tmp on the existing pod, then starts the
in-pod training script in a tmux session.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True)


def kubectl_exec(namespace: str, pod: str, remote_cmd: str) -> None:
    run(["kubectl", "exec", "-n", namespace, pod, "--", "bash", "-lc", remote_cmd])


def quote(value: str) -> str:
    return shlex.quote(value)


def prepare_project(namespace: str, pod: str, project_dir: str, branch: str, repo_url: str) -> None:
    remote_cmd = f"""
set -Eeuo pipefail
if [[ ! -d {quote(project_dir)}/.git ]]; then
  rm -rf {quote(project_dir)}
  git clone --branch {quote(branch)} {quote(repo_url)} {quote(project_dir)}
else
  cd {quote(project_dir)}
  git fetch origin {quote(branch)}
  git checkout {quote(branch)}
  git reset --hard origin/{quote(branch)}
fi
cd {quote(project_dir)}
git status --short --branch
git rev-parse --short HEAD
"""
    kubectl_exec(namespace, pod, remote_cmd)


def start_tmux(
    namespace: str,
    pod: str,
    project_dir: str,
    session: str,
    task_name: str,
    replace: bool,
    smoke: bool,
) -> None:
    log_dir = f"/mnt/nuplan/projects/catk/logs/tmux_static_multinode/{task_name}"
    log_file = f"{log_dir}/{pod}.road_flow.log"

    env = {
        "TASK_NAME": task_name,
        "CACHE_ROOT": "/workspace/womd_v1_3/SMART_cache",
        "TRAIN_BATCH_SIZE": "12",
        "VAL_BATCH_SIZE": "12",
        "TEST_BATCH_SIZE": "12",
        "NPROC_PER_NODE": "7",
        "TRAINER_DEVICES": "7",
        "CKPT_ARTIFACT": "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57",
        "CKPT_DOWNLOAD_DIR": "/workspace/flow_control_space_pretrain_x5f9g0ce/v57",
        "ROAD_WORK_DIR": f"/workspace/road_cache/{task_name}",
        "ROAD_GENERATION_BATCH_SIZE": "8",
        "ROAD_CANDIDATE_MICRO_BATCH_SIZE": "16",
    }
    if smoke:
        env.update(
            {
                "TASK_NAME": task_name,
                "ROAD_DATA_USE_RATIO": "0.000005",
                "MAX_EPOCHS": "2",
                "LIMIT_TRAIN_BATCHES": "1",
                "LIMIT_VAL_BATCHES": "0",
                "ROAD_CLEANUP_USED_CACHE": "false",
                "ROAD_OVERWRITE_CACHE": "true",
            }
        )

    exports = "\n".join(f"export {key}={quote(value)}" for key, value in env.items())
    kill_existing = f"tmux kill-session -t {quote(session)} 2>/dev/null || true" if replace else ""
    remote_cmd = f"""
set -Eeuo pipefail
mkdir -p {quote(log_dir)}
{kill_existing}
if tmux has-session -t {quote(session)} 2>/dev/null; then
  echo "tmux session already exists: {session}" >&2
  exit 3
fi
tmux new-session -d -s {quote(session)} "cd {quote(project_dir)} && {exports}; bash scripts/start_road_flow_testas_a100x7_x5f9g0ce.sh 2>&1 | tee {quote(log_file)}"
echo "session={session}"
echo "log={log_file}"
tmux ls | grep {quote(session)}
"""
    kubectl_exec(namespace, pod, remote_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="p-pnc")
    parser.add_argument("--pod", default="testas")
    parser.add_argument("--branch", default="self_forcing_w_road")
    parser.add_argument("--repo-url", default="https://github.com/seulbinHwang/catk.git")
    parser.add_argument("--project-dir", default="/tmp/catk_self_forcing_w_road_road_flow_testas")
    parser.add_argument("--session", default="catk-road-flow-testas-a100x7-self-forcing-w-road")
    parser.add_argument(
        "--task-name",
        default="road_flow_a100x7_testas_self_forcing_w_road_x5f9g0ce_epoch061_bs12",
    )
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny cache-generation/training smoke job instead of the full fine-tune.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_name = args.task_name
    if args.smoke and task_name == "road_flow_a100x7_testas_self_forcing_w_road_x5f9g0ce_epoch061_bs12":
        task_name = f"road_flow_a100x7_testas_self_forcing_w_road_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    prepare_project(args.namespace, args.pod, args.project_dir, args.branch, args.repo_url)
    start_tmux(
        args.namespace,
        args.pod,
        args.project_dir,
        args.session,
        task_name,
        args.replace,
        args.smoke,
    )


if __name__ == "__main__":
    main()
