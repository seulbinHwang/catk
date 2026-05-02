#!/usr/bin/env python3
"""Prepare CAT-K repo, Python env, cache, and checkpoint on static MLX pods."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
import time


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["testsv", "testsvv", "testsvvv", "testsvvvv"]
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_continuous_track_loss"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_CACHE_SOURCE = "labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache"
DEFAULT_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-4pxhrpv8:v70"
DEFAULT_CKPT_PATH = (
    "/mnt/nuplan/projects/catk/checkpoints/"
    "flow_semi_continuous_pretrain_all_target_h1006/"
    "4pxhrpv8_v70_e64_step259776/epoch_last.ckpt"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout else ""


def kubectl_exec(namespace: str, pod: str, container: str, script: str, *, check: bool = True) -> str:
    return run(
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
        capture=True,
        check=check,
    )


def log(message: str) -> None:
    print(f"[{dt.datetime.now():%F %T}] {message}", flush=True)


def remote_paths(args: argparse.Namespace, pod: str) -> tuple[str, str, str, str]:
    root = f"{args.project_root.rstrip('/')}/logs/static_pod_prepare"
    return (
        root,
        f"{root}/{pod}.log",
        f"{root}/{pod}.done",
        f"{root}/{pod}.failed",
    )


def render_prepare_script(args: argparse.Namespace, pod: str) -> str:
    root, log_path, done_path, failed_path = remote_paths(args, pod)
    install_block = ""
    if not args.skip_requirements:
        install_block = """\
echo "[prepare] installing README Python requirements"
python -m pip install --upgrade pip
python -m pip install -r install/requirements.txt
python -m pip install torch_geometric
python -m pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
"""
    cache_block = ""
    if not args.skip_cache:
        cache_block = f"""\
export CACHE_ROOT={shq(args.cache_root)}
mkdir -p "$(dirname "$CACHE_ROOT")"
if [ -d "$CACHE_ROOT" ] && [ -n "$(find "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "[prepare] cache already exists at $CACHE_ROOT"
else
  echo "[prepare] downloading SMART cache to $CACHE_ROOT"
  bash scripts/download_smart_cache_from_nubes.sh {shq(args.cache_source)} "$CACHE_ROOT"
fi
"""
    ckpt_block = ""
    if not args.skip_checkpoint:
        ckpt_block = f"""\
CKPT_PATH={shq(args.ckpt_path)}
ARTIFACT={shq(args.artifact)}
if [ -s "$CKPT_PATH" ]; then
  echo "[prepare] checkpoint already exists: $CKPT_PATH"
else
  echo "[prepare] downloading checkpoint artifact $ARTIFACT"
  mkdir -p "$(dirname "$CKPT_PATH")"
  TMP_ARTIFACT_DIR={shq(args.project_root.rstrip('/') + '/checkpoints/.wandb_epoch-last-4pxhrpv8_v70')}
  rm -rf "$TMP_ARTIFACT_DIR"
  mkdir -p "$TMP_ARTIFACT_DIR"
  if [ -n "${{WANDB_API_KEY:-}}" ]; then
    wandb login --relogin "$WANDB_API_KEY" || true
  fi
  wandb artifact get "$ARTIFACT" --root "$TMP_ARTIFACT_DIR"
  FOUND="$(find "$TMP_ARTIFACT_DIR" -type f -name epoch_last.ckpt | head -1)"
  if [ -z "$FOUND" ]; then
    echo "[prepare] epoch_last.ckpt not found inside $TMP_ARTIFACT_DIR" >&2
    find "$TMP_ARTIFACT_DIR" -maxdepth 4 -type f >&2 || true
    exit 1
  fi
  cp "$FOUND" "$CKPT_PATH"
fi
"""
    verify_cache = "" if args.skip_cache else f"""\
test -d {shq(args.cache_root)}
find {shq(args.cache_root)} -mindepth 1 -maxdepth 2 -print -quit | grep -q .
"""
    verify_ckpt = "" if args.skip_checkpoint else f"test -s {shq(args.ckpt_path)}\n"
    verify_python = "" if args.skip_requirements else """\
python - <<'PY'
import lightning
import torch
import torch_cluster
import torch_geometric
import torch_scatter
import waymo_open_dataset
print("python imports ok", torch.__version__, lightning.__version__)
PY
"""
    return f"""set -Eeuo pipefail
mkdir -p {shq(root)}
rm -f {shq(done_path)} {shq(failed_path)}
exec > >(tee -a {shq(log_path)}) 2>&1
trap 'echo "[prepare] FAILED at $(date "+%F %T")"; touch {shq(failed_path)}' ERR

echo "[prepare] pod={pod} started at $(date '+%F %T')"
export PROJECT_ROOT={shq(args.project_root)}

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  echo "[prepare] cloning catk repo"
  mkdir -p "$(dirname "$PROJECT_ROOT")"
  git clone https://github.com/seulbinHwang/catk.git "$PROJECT_ROOT"
fi

git config --global --add safe.directory "$PROJECT_ROOT" || true
cd "$PROJECT_ROOT"
git fetch origin {shq(args.branch)}
git checkout {shq(args.branch)} 2>/dev/null || git checkout -b {shq(args.branch)} origin/{args.branch}
git pull --ff-only origin {shq(args.branch)}

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
else
  echo "[prepare] conda not found at /mnt/nuplan/miniforge" >&2
  exit 1
fi

{install_block.rstrip()}
{cache_block.rstrip()}
{ckpt_block.rstrip()}

echo "[prepare] verification"
{verify_cache.rstrip()}
{verify_ckpt.rstrip()}
{verify_python.rstrip()}
python -m py_compile src/smart/model/smart_flow.py
bash -n scripts/mlx_finetune_draft_flow_v100x8_multinode.sh
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

touch {shq(done_path)}
rm -f {shq(failed_path)}
echo "[prepare] DONE at $(date '+%F %T')"
"""


def start_prepare(args: argparse.Namespace, pod: str) -> None:
    root, log_path, done_path, failed_path = remote_paths(args, pod)
    prepare_script = render_prepare_script(args, pod)
    replace_block = ""
    if args.replace:
        replace_block = f"tmux kill-session -t {shq(args.session)} 2>/dev/null || true"
    script = f"""set -Eeuo pipefail
mkdir -p {shq(root)}
{replace_block}
cat > {shq(root + '/' + pod + '_prepare.sh')} <<'CATK_PREPARE'
{prepare_script.rstrip()}
CATK_PREPARE
chmod +x {shq(root + '/' + pod + '_prepare.sh')}
rm -f {shq(done_path)} {shq(failed_path)}
: > {shq(log_path)}
tmux new-session -d -s {shq(args.session)} {shq(root + '/' + pod + '_prepare.sh')}
echo {shq(log_path)}
"""
    out = kubectl_exec(args.namespace, pod, args.container, script)
    log(f"started prepare tmux on {pod}: {out}")


def status(args: argparse.Namespace, pod: str) -> tuple[str, str]:
    _, log_path, done_path, failed_path = remote_paths(args, pod)
    script = f"""set +e
if [ -f {shq(done_path)} ]; then echo DONE; exit 0; fi
if [ -f {shq(failed_path)} ]; then echo FAILED; tail -n 80 {shq(log_path)} 2>/dev/null; exit 0; fi
if tmux has-session -t {shq(args.session)} 2>/dev/null; then echo RUNNING; tail -n 12 {shq(log_path)} 2>/dev/null; exit 0; fi
echo UNKNOWN
tail -n 80 {shq(log_path)} 2>/dev/null
"""
    text = kubectl_exec(args.namespace, pod, args.container, script, check=False)
    first, _, rest = text.partition("\n")
    return first.strip(), rest.strip()


def wait_all(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.timeout
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        states = {pod: status(args, pod) for pod in args.pods}
        if all(state == "DONE" for state, _ in states.values()):
            log("all pods prepared")
            return
        failed = {pod: detail for pod, (state, detail) in states.items() if state in {"FAILED", "UNKNOWN"}}
        if failed:
            for pod, detail in failed.items():
                log(f"{pod} prepare failed/unknown:\n{detail}")
            raise SystemExit("pod preparation failed")
        now = time.monotonic()
        if now - last_heartbeat >= args.heartbeat_interval:
            for pod, (state, detail) in states.items():
                tail = ("\n" + detail) if detail else ""
                log(f"{pod}: {state}{tail}")
            last_heartbeat = now
        time.sleep(args.poll_interval)
    raise TimeoutError(f"pod preparation did not finish within {args.timeout}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default="main")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache-source", default=DEFAULT_CACHE_SOURCE)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--session", default="catk-pod-prepare")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-requirements", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--heartbeat-interval", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status_only:
        for pod in args.pods:
            state, detail = status(args, pod)
            print(f"{pod}: {state}")
            if detail:
                print(detail)
        return
    for pod in args.pods:
        start_prepare(args, pod)
    if not args.start_only:
        wait_all(args)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, TimeoutError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
