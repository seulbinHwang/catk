#!/usr/bin/env python3
"""Launch UniMM Anchor-Based-4s on hsb-npc-training-3-{1,2} H100 pods.

The launcher avoids touching the dirty shared checkout under
``/mnt/nuplan/projects/catk`` by preparing a clean checkout in ``/tmp`` on each
pod. It never creates, deletes, or restarts Kubernetes pods, and only manages
the tmux session name passed to this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import posixpath
import shlex
import subprocess
import sys
import tempfile


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["hsb-npc-training-3-1", "hsb-npc-training-3-2"]
DEFAULT_BRANCH = "UniMM"
DEFAULT_REPO_URL = "https://github.com/seulbinHwang/catk.git"
DEFAULT_PROJECT_ROOT = "/tmp/catk_unimm_h100x3x2"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_ANCHOR_FILE = ""
DEFAULT_EXPECTED_GPUS = 3
DEFAULT_LEARNING_RATE = "0.001224744871"
KUBECTL_BIN = os.environ.get("KUBECTL_BIN") or (
    "/usr/local/bin/kubectl" if os.path.exists("/usr/local/bin/kubectl") else "kubectl"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def env_line(name: str, value: object) -> str:
    return f"{name}={shq(value)}"


def run_kubectl(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        [KUBECTL_BIN, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def exec_capture_in_pod(namespace: str, container: str, pod: str, script: str) -> str:
    return run_kubectl(
        ["exec", "-n", namespace, pod, "-c", container, "--", "bash", "-lc", script],
        capture=True,
    )


def pod_ip(namespace: str, pod: str) -> str:
    return run_kubectl(
        ["get", "pod", pod, "-n", namespace, "-o", "jsonpath={.status.podIP}"],
        capture=True,
    )


def pod_gpu_count(namespace: str, container: str, pod: str) -> int:
    output = run_kubectl(
        [
            "exec",
            "-n",
            namespace,
            pod,
            "-c",
            container,
            "--",
            "bash",
            "-lc",
            "nvidia-smi -L 2>/dev/null | wc -l",
        ],
        capture=True,
    )
    return int(output.strip())


def exec_in_pod(
    namespace: str,
    container: str,
    pod: str,
    script: str,
    *,
    dry_run: bool,
) -> None:
    cmd = ["exec", "-n", namespace, pod, "-c", container, "--", "bash", "-lc", script]
    if dry_run:
        print("kubectl " + " ".join(shq(part) for part in cmd))
        return
    run_kubectl(cmd)


def safe_remote_name(value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=6).hexdigest()
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    safe = safe.strip("._-") or "task"
    return f"{safe[:80]}_{digest}"


def checkpoint_sync_path(args: argparse.Namespace) -> str:
    basename = posixpath.basename(args.ckpt_path.rstrip("/")) or "checkpoint.ckpt"
    return posixpath.join(
        "/tmp/unimm_h100x3x2_synced_ckpts",
        safe_remote_name(args.task_name),
        basename,
    )


def remote_sha256(namespace: str, container: str, pod: str, path: str) -> str:
    return exec_capture_in_pod(
        namespace,
        container,
        pod,
        f"sha256sum {shq(path)} | awk '{{print $1}}'",
    ).strip()


def copy_file_between_pods(
    namespace: str,
    container: str,
    src_pod: str,
    dst_pod: str,
    src_path: str,
    dst_path: str,
) -> None:
    dst_dir = posixpath.dirname(dst_path) or "."
    local_name = posixpath.basename(dst_path.rstrip("/")) or "checkpoint.ckpt"
    with tempfile.TemporaryDirectory(prefix="unimm_ckpt_sync_") as tmp_dir:
        local_path = os.path.join(tmp_dir, local_name)
        subprocess.run(
            [
                KUBECTL_BIN,
                "cp",
                "-n",
                namespace,
                "-c",
                container,
                f"{src_pod}:{src_path}",
                local_path,
            ],
            check=True,
        )
        exec_in_pod(
            namespace,
            container,
            dst_pod,
            f"mkdir -p {shq(dst_dir)}",
            dry_run=False,
        )
        subprocess.run(
            [
                KUBECTL_BIN,
                "cp",
                "-n",
                namespace,
                "-c",
                container,
                local_path,
                f"{dst_pod}:{dst_path}",
            ],
            check=True,
        )


def sync_checkpoint_to_pods(args: argparse.Namespace) -> None:
    if not args.ckpt_path or args.no_sync_ckpt:
        return

    source_pod = args.pods[0]
    original_path = args.ckpt_path
    synced_path = checkpoint_sync_path(args)
    sync_dir = posixpath.dirname(synced_path) or "."
    if args.dry_run:
        print(
            "[launcher] dry-run checkpoint sync: "
            f"{source_pod}:{original_path} -> all pods:{synced_path}",
            flush=True,
        )
        args.ckpt_path = synced_path
        return

    print(
        "[launcher] syncing checkpoint for multi-node access: "
        f"{source_pod}:{original_path} -> {synced_path}",
        flush=True,
    )
    master_script = f"""
set -Eeuo pipefail
src={shq(original_path)}
dst={shq(synced_path)}
if [ ! -f "$src" ]; then
  echo "[launcher] checkpoint not found on source pod {source_pod}: $src" >&2
  exit 2
fi
mkdir -p {shq(sync_dir)}
if [ "$(readlink -f "$src")" != "$(readlink -f "$dst" 2>/dev/null || true)" ]; then
  cp -f --dereference "$src" "$dst"
fi
sha256sum "$dst" | awk '{{print $1}}'
"""
    source_sha = exec_capture_in_pod(
        args.namespace,
        args.container,
        source_pod,
        master_script,
    ).strip().splitlines()[-1]
    if not source_sha:
        raise RuntimeError(f"failed to compute checkpoint sha256 on {source_pod}:{synced_path}")

    for pod in args.pods[1:]:
        print(f"[launcher] copying checkpoint to {pod}:{synced_path}", flush=True)
        copy_file_between_pods(
            args.namespace,
            args.container,
            source_pod,
            pod,
            synced_path,
            synced_path,
        )
        pod_sha = remote_sha256(args.namespace, args.container, pod, synced_path)
        if pod_sha != source_sha:
            raise RuntimeError(
                "checkpoint sync verification failed: "
                f"{source_pod}:{source_sha} != {pod}:{pod_sha}"
            )

    args.ckpt_path = synced_path
    print(f"[launcher] checkpoint sync complete: ckpt_path={args.ckpt_path}", flush=True)


def prepare_checkout_block(args: argparse.Namespace) -> str:
    return f"""
if [ ! -d {shq(args.project_root)}/.git ]; then
  rm -rf {shq(args.project_root)}
  git clone {shq(args.repo_url)} {shq(args.project_root)}
fi
cd {shq(args.project_root)}
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin --prune
if git show-ref --verify --quiet refs/heads/{shq(args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} origin/{shq(args.branch)}
fi
git reset --hard origin/{shq(args.branch)}
git clean -fdx
"""


def stop_command(session: str, task_name: str) -> str:
    return f"""set +e
task_name={shq(task_name)}
session={shq(session)}
mapfile -t pids < <(
  ps -eo pid=,cmd= |
    awk -v task="task_name=${{task_name}}" '
      $0 ~ task && ($0 ~ /(^|[ /])python([0-9.]*)?([[:space:]]|$)/ || $0 ~ /(^|[ /])torchrun([[:space:]]|$)/) {{ print $1 }}
    ' |
    while read -r pid; do
      if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
        echo "$pid"
      fi
    done
)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating task processes for $task_name: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep "${{REMOTE_KILL_GRACE_SEC:-20}}"
  mapfile -t pids < <(
    ps -eo pid=,cmd= |
      awk -v task="task_name=${{task_name}}" '
        $0 ~ task && ($0 ~ /(^|[ /])python([0-9.]*)?([[:space:]]|$)/ || $0 ~ /(^|[ /])torchrun([[:space:]]|$)/) {{ print $1 }}
      ' |
      while read -r pid; do
        if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
          echo "$pid"
        fi
      done
  )
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing task processes for $task_name: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
if tmux has-session -t "$session" 2>/dev/null; then
  tmux kill-session -t "$session"
  echo "[launcher] stopped tmux session $session"
else
  echo "[launcher] tmux session not found: $session"
fi
"""


def build_anchors_command(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    log_root = f"{args.log_dir.rstrip('/')}/tmux_unimm_h100x3x2/{safe_task}"
    run_file = f"{log_root}/build_anchors.sh"
    log_file = f"{log_root}/build_anchors.tmux.log"
    script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {shq(args.project_root)}
export CONDA_ROOT={shq(args.conda_root)}
export CACHE_ROOT={shq(args.cache_root)}
export OUTPUT={shq(args.anchor_file)}
mkdir -p "$(dirname "$OUTPUT")"
bash scripts/build_unimm_anchors.sh --device {shq(args.anchor_device)}
"""
    return f"""set -Eeuo pipefail
{prepare_checkout_block(args)}
mkdir -p {shq(log_root)}
cat > {shq(run_file)} <<'UNIMM_BUILD'
{script.rstrip()}
UNIMM_BUILD
chmod +x {shq(run_file)}
: > {shq(log_file)}
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq("cat >> " + log_file)}
echo "[launcher] started anchor build tmux session {args.session}"
echo "[launcher] log: {log_file}"
"""


def launch_command(
    args: argparse.Namespace,
    pod: str,
    node_rank: int,
    master_addr: str,
    gpu_count: int,
) -> str:
    safe_task = args.task_name.replace("/", "_")
    log_root = f"{args.log_dir.rstrip('/')}/tmux_unimm_h100x3x2/{safe_task}"
    env_file = f"{log_root}/{pod}.env"
    run_file = f"{log_root}/{pod}_run.sh"
    log_file = f"{log_root}/{pod}.tmux.log"
    status_file = f"{log_root}/{pod}.torchrun_status"
    pgid_file = f"{log_root}/{pod}.torchrun_pgid"
    hydra_overrides = args.extra_hydra_overrides
    if args.smoke:
        hydra_overrides = " ".join(
            [
                hydra_overrides,
                "trainer.max_epochs=1",
                f"trainer.limit_train_batches={args.smoke_batches}",
                "trainer.limit_val_batches=0",
                "model.model_config.val_open_loop=false",
                "model.model_config.val_closed_loop=false",
                "logger.wandb.offline=true",
                "logger.wandb.log_model=false",
            ]
        ).strip()
    env_text = "\n".join(
        [
            env_line("CONDA_ROOT", args.conda_root),
            env_line("CACHE_ROOT", args.cache_root),
            env_line("UNIMM_ANCHOR_FILE", args.anchor_file),
            env_line("NNODES", len(args.pods)),
            env_line("NPROC_PER_NODE", gpu_count),
            env_line("TRAINER_DEVICES", gpu_count),
            env_line("NODE_RANK", node_rank),
            env_line("MANUAL_RANK_OFFSET", node_rank * gpu_count),
            env_line("MANUAL_WORLD_SIZE", len(args.pods) * gpu_count),
            env_line("MASTER_ADDR", master_addr),
            env_line("MASTER_PORT", args.master_port),
            env_line("TASK_NAME", args.task_name),
            env_line("CATK_ACTION", args.action),
            env_line("LOG_DIR", args.log_dir),
            env_line("TRAIN_BATCH_SIZE", args.train_batch_size),
            env_line("VAL_BATCH_SIZE", args.val_batch_size),
            env_line("TEST_BATCH_SIZE", args.test_batch_size),
            env_line("LEARNING_RATE", args.learning_rate),
            env_line("WANDB_MODE", args.wandb_mode),
            env_line("CKPT_PATH", args.ckpt_path),
            env_line("LIMIT_TRAIN_BATCHES", args.limit_train_batches),
            env_line("LIMIT_VAL_BATCHES", args.limit_val_batches),
            env_line("MAX_EPOCHS", args.max_epochs),
            env_line("CATK_HYDRA_OVERRIDES", hydra_overrides),
        ]
    ) + "\n"
    run_text = f"""#!/usr/bin/env bash
set +e
cd {shq(args.project_root)}
set -a
source {shq(env_file)}
set +a
rm -f {shq(status_file)}
ps -o pgid= -p "$$" | tr -d '[:space:]' > {shq(pgid_file)} 2>/dev/null || true
echo "[tmux-run] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
bash scripts/unimm_h100x3x2_train.sh
status=$?
echo "$status" > {shq(status_file)}
echo "[tmux-run] exited with status $status at $(date '+%F %T')"
exec bash
"""
    replace_block = ""
    if args.replace:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
"""
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  exit 3
fi
"""
    return f"""set -Eeuo pipefail
{prepare_checkout_block(args)}
if [ ! -f {shq(args.anchor_file)} ]; then
  echo "[launcher] anchor file is missing: {args.anchor_file}" >&2
  echo "[launcher] run with --build-anchors first or set --anchor-file." >&2
  exit 2
fi
{replace_block}
mkdir -p {shq(log_root)}
cat > {shq(env_file)} <<'UNIMM_ENV'
{env_text.rstrip()}
UNIMM_ENV
cat > {shq(run_file)} <<'UNIMM_RUN'
{run_text.rstrip()}
UNIMM_RUN
chmod +x {shq(run_file)}
: > {shq(log_file)}
rm -f {shq(status_file)} {shq(pgid_file)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq("cat >> " + log_file)}
echo "[launcher] started tmux session {args.session} on pod {pod}"
echo "[launcher] log: {log_file}"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch UniMM Anchor-Based-4s on hsb-npc-training-3-{1,2} H100 pods.",
    )
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE))
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "main"))
    parser.add_argument("--pods", nargs="+", default=os.environ.get("PODS", " ".join(DEFAULT_PODS)).split())
    parser.add_argument("--repo-url", default=os.environ.get("CATK_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--branch", default=os.environ.get("CATK_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--project-root", default=os.environ.get("REMOTE_PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--log-dir", default=os.environ.get("REMOTE_LOG_DIR", DEFAULT_LOG_DIR))
    parser.add_argument("--anchor-file", default=os.environ.get("UNIMM_ANCHOR_FILE", DEFAULT_ANCHOR_FILE))
    parser.add_argument("--conda-root", default=os.environ.get("CONDA_ROOT", "/mnt/nuplan/miniforge"))
    parser.add_argument("--action", choices=["fit", "validate", "test"], default="fit")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--session", default="unimm-h100x3x2")
    parser.add_argument("--master-port", default="29551")
    parser.add_argument("--expected-gpus", type=int, default=DEFAULT_EXPECTED_GPUS)
    parser.add_argument("--train-batch-size", default="32")
    parser.add_argument("--val-batch-size", default="12")
    parser.add_argument("--test-batch-size", default="4")
    parser.add_argument("--learning-rate", default=os.environ.get("UNIMM_LEARNING_RATE", DEFAULT_LEARNING_RATE))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--ckpt-path", default=os.environ.get("CKPT_PATH", ""))
    parser.add_argument("--limit-train-batches", default=os.environ.get("LIMIT_TRAIN_BATCHES", ""))
    parser.add_argument("--limit-val-batches", default=os.environ.get("LIMIT_VAL_BATCHES", ""))
    parser.add_argument("--max-epochs", default=os.environ.get("MAX_EPOCHS", ""))
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument(
        "--no-sync-ckpt",
        action="store_true",
        help=(
            "Do not copy --ckpt-path into a same-path checkpoint file on every "
            "pod. Only use this when the path is truly shared across all nodes."
        ),
    )
    parser.add_argument("--anchor-device", default="cuda")
    parser.add_argument("--build-anchors", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-batches", default="2")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.anchor_file:
        args.anchor_file = f"{args.project_root.rstrip('/')}/src/unimm/anchors/unimm_anchors_8s_k2048.pkl"
    if len(args.pods) != 2 and not args.stop and not args.build_anchors:
        parser.error("default H100 recipe expects exactly two pods")
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "smoke" if args.smoke else "train"
        args.task_name = f"unimm_anchor_based_4s_h100x3x2_{suffix}_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        for pod in args.pods:
            exec_in_pod(
                args.namespace,
                args.container,
                pod,
                stop_command(args.session, args.task_name),
                dry_run=args.dry_run,
            )
        return

    if args.build_anchors:
        exec_in_pod(
            args.namespace,
            args.container,
            args.pods[0],
            build_anchors_command(args),
            dry_run=args.dry_run,
        )
        return

    master_addr = "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    gpu_counts: dict[str, int] = {}
    for pod in args.pods:
        gpu_counts[pod] = args.expected_gpus if args.dry_run else pod_gpu_count(args.namespace, args.container, pod)
        if gpu_counts[pod] != args.expected_gpus:
            raise RuntimeError(f"expected {args.expected_gpus} GPUs in {pod}, found {gpu_counts[pod]}")

    sync_checkpoint_to_pods(args)

    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})", flush=True)
    print(f"[launcher] task_name: {args.task_name}", flush=True)
    print(f"[launcher] anchor:    {args.anchor_file}", flush=True)
    for node_rank, pod in enumerate(args.pods):
        exec_in_pod(
            args.namespace,
            args.container,
            pod,
            launch_command(args, pod, node_rank, master_addr, gpu_counts[pod]),
            dry_run=args.dry_run,
        )

    print("\nAttach commands:")
    for pod in args.pods:
        print(
            "  kubectl exec -it "
            f"-n {args.namespace} {pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
