#!/usr/bin/env python3
"""Launch DRaFT fine-tuning on existing V100x4x8 static pods.

This launcher never creates, deletes, or restarts pods. It only uses
``kubectl exec`` to start or stop a tmux session inside already-running pods.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = [
    "testsv",
    "testsvv",
    "testsvvv",
    "testsvvvv",
    "sv",
    "svv",
    "svvv",
    "svvvv",
]
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing_w_track_loss"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "finetune_draft_flow_v100x4x8"
DEFAULT_TASK_NAME = "flow_finetune_draft_v100x4x8_bs24_soft08_topk20_commit1_noslip"
DEFAULT_SESSION = "catk-draft-v100x4x8-bs24-soft08"


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def pod_ip(namespace: str, pod: str) -> str:
    return run_kubectl(
        ["get", "pod", pod, "-n", namespace, "-o", "jsonpath={.status.podIP}"],
        capture=True,
    )


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_static_multinode/{safe_task}"


def render_env(args: argparse.Namespace, *, rank: int, master_addr: str) -> str:
    lines = [
        export_line("PROJECT_ROOT", args.project_root),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("CKPT_PATH", args.ckpt_path),
        export_line("WANDB_ARTIFACT", args.wandb_artifact),
        export_line("ARTIFACT_DOWNLOAD_DIR", args.artifact_download_dir),
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("NNODES", len(args.pods)),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("NODE_RANK", rank),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("CHECKPOINT_SYNC_HOST", master_addr),
        export_line("CHECKPOINT_SYNC_PORT", args.checkpoint_sync_port),
        export_line("LOG_DIR", args.log_dir),
        export_line("RUN_ROOT", run_root(args)),
        export_line("TRAIN_BATCH_SIZE", args.train_batch_size),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("PRECISION", args.precision),
        export_line("CATK_LR", args.learning_rate),
        export_line("CATK_EXTRA_OVERRIDES", args.extra_hydra_overrides),
    ]
    return "\n".join(lines) + "\n"


def render_worker_script(env_file: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${{TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}}"
export TORCH_NCCL_BLOCKING_WAIT="${{TORCH_NCCL_BLOCKING_WAIT:-0}}"

if [ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk 2>/dev/null || conda activate base 2>/dev/null || true
fi

set -a
source {shq(env_file)}
set +a

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"

echo "[draft-v100x4x8] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[draft-v100x4x8] started at $(date '+%F %T')"
echo "[draft-v100x4x8] experiment=${{EXPERIMENT}} bs=${{TRAIN_BATCH_SIZE}} precision=${{PRECISION}}"
echo "[draft-v100x4x8] ckpt_path=${{CKPT_PATH}}"
echo "[draft-v100x4x8] attach survives after exit; press Ctrl-b d to detach"
echo

CHECKPOINT_SYNC_PID=""

start_checkpoint_sync_server() {{
  if (( NODE_RANK != 0 )); then
    return 0
  fi
  python - <<'PY' &
import hashlib
import http.server
import os
import pathlib

ckpt_path = pathlib.Path(os.environ["CKPT_PATH"])
port = int(os.environ["CHECKPOINT_SYNC_PORT"])


def checkpoint_metadata():
    stat = ckpt_path.stat()
    digest = hashlib.sha256()
    with ckpt_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return stat.st_size, digest.hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_text(self, code, text):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send_text(200, "ok\\n")
            return
        if not ckpt_path.is_file():
            self.send_text(404, "missing checkpoint\\n")
            return
        if path == "/metadata":
            size, sha = checkpoint_metadata()
            self.send_text(200, f"size={{size}}\\nsha256={{sha}}\\npath={{ckpt_path}}\\n")
            return
        if path == "/checkpoint":
            size = ckpt_path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with ckpt_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    self.wfile.write(chunk)
            return
        self.send_text(404, "not found\\n")


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


Server(("", port), Handler).serve_forever()
PY
  CHECKPOINT_SYNC_PID=$!
  echo "[draft-v100x4x8] checkpoint sync server started on $CHECKPOINT_SYNC_HOST:$CHECKPOINT_SYNC_PORT pid=$CHECKPOINT_SYNC_PID"
}}

stop_checkpoint_sync_server() {{
  if [[ -n "$CHECKPOINT_SYNC_PID" ]]; then
    kill "$CHECKPOINT_SYNC_PID" 2>/dev/null || true
  fi
}}

wait_for_checkpoint_sync_server() {{
  local waited=0
  local timeout_sec="${{CHECKPOINT_SYNC_START_TIMEOUT_SEC:-600}}"
  while true; do
    if python - <<'PY' >/dev/null 2>&1
import os
import urllib.request

url = "http://" + os.environ["CHECKPOINT_SYNC_HOST"] + ":" + os.environ["CHECKPOINT_SYNC_PORT"] + "/health"
with urllib.request.urlopen(url, timeout=5) as response:
    response.read()
PY
    then
      return 0
    fi
    if (( waited >= timeout_sec )); then
      echo "[draft-v100x4x8] timed out waiting for checkpoint sync server at $CHECKPOINT_SYNC_HOST:$CHECKPOINT_SYNC_PORT" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
}}

fetch_checkpoint_metadata() {{
  python - <<'PY'
import os
import sys
import urllib.request

url = "http://" + os.environ["CHECKPOINT_SYNC_HOST"] + ":" + os.environ["CHECKPOINT_SYNC_PORT"] + "/metadata"
with urllib.request.urlopen(url, timeout=30) as response:
    sys.stdout.write(response.read().decode("utf-8"))
PY
}}

metadata_value() {{
  local metadata="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key {{ print substr($0, index($0, "=") + 1); exit }}' <<< "$metadata"
}}

checkpoint_matches() {{
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local actual_size=""
  local actual_sha=""

  [[ -f "$path" ]] || return 1
  actual_size="$(stat -c %s "$path" 2>/dev/null || true)"
  [[ "$actual_size" == "$expected_size" ]] || return 1
  actual_sha="$(sha256sum "$path" 2>/dev/null | awk '{{ print $1 }}')"
  [[ "$actual_sha" == "$expected_sha" ]]
}}

download_synced_checkpoint() {{
  local metadata="$1"
  local expected_size=""
  local expected_sha=""
  local tmp_file=""

  expected_size="$(metadata_value "$metadata" size)"
  expected_sha="$(metadata_value "$metadata" sha256)"
  if [[ -z "$expected_size" || -z "$expected_sha" ]]; then
    echo "[draft-v100x4x8] invalid checkpoint metadata from rank0" >&2
    return 1
  fi
  if checkpoint_matches "$CKPT_PATH" "$expected_size" "$expected_sha"; then
    echo "[draft-v100x4x8] checkpoint already synced: $CKPT_PATH"
    return 0
  fi

  mkdir -p "$(dirname "$CKPT_PATH")"
  tmp_file="${{CKPT_PATH}}.download.$$"
  rm -f "$tmp_file"
  echo "[draft-v100x4x8] downloading rank0 checkpoint to $CKPT_PATH"
  if ! python - "$tmp_file" <<'PY'
import os
import shutil
import sys
import urllib.request

target = sys.argv[1]
url = "http://" + os.environ["CHECKPOINT_SYNC_HOST"] + ":" + os.environ["CHECKPOINT_SYNC_PORT"] + "/checkpoint"
with urllib.request.urlopen(url, timeout=120) as response:
    with open(target, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
PY
  then
    rm -f "$tmp_file"
    return 1
  fi
  if ! checkpoint_matches "$tmp_file" "$expected_size" "$expected_sha"; then
    echo "[draft-v100x4x8] downloaded checkpoint failed verification: $tmp_file" >&2
    rm -f "$tmp_file"
    return 1
  fi
  mv -f "$tmp_file" "$CKPT_PATH"
  checkpoint_matches "$CKPT_PATH" "$expected_size" "$expected_sha"
}}

ensure_checkpoint_local() {{
  if [[ -f "$CKPT_PATH" ]]; then
    echo "[draft-v100x4x8] using checkpoint: $CKPT_PATH"
    return 0
  fi
  if [[ -z "$WANDB_ARTIFACT" ]]; then
    echo "[draft-v100x4x8] ERROR: checkpoint not found and WANDB_ARTIFACT is empty: $CKPT_PATH" >&2
    return 2
  fi

  local download_dir="${{ARTIFACT_DOWNLOAD_DIR:-$(dirname "$CKPT_PATH")/artifact}}"
  mkdir -p "$(dirname "$CKPT_PATH")" "$download_dir"
  local lock_dir="${{CKPT_PATH}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[draft-v100x4x8] downloading W&B artifact: $WANDB_ARTIFACT"
    python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_ARTIFACT"]
download_dir = os.environ["ARTIFACT_DOWNLOAD_DIR"] or str(Path(os.environ["CKPT_PATH"]).parent / "artifact")
target_ckpt = os.environ["CKPT_PATH"]

try:
    import wandb
except Exception as exc:
    print(f"ERROR: failed to import wandb: {{exc}}", file=sys.stderr)
    sys.exit(2)

Path(download_dir).mkdir(parents=True, exist_ok=True)
Path(target_ckpt).parent.mkdir(parents=True, exist_ok=True)

api = wandb.Api()
artifact = api.artifact(artifact_name)
artifact_dir = artifact.download(root=download_dir)

candidates = []
preferred = Path(artifact_dir) / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred.as_posix())
candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(glob.glob(str(Path(artifact_dir) / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))

if not candidates:
    print(f"ERROR: no checkpoint file found in artifact dir: {{artifact_dir}}", file=sys.stderr)
    sys.exit(3)

source = candidates[0]
if os.path.abspath(source) != os.path.abspath(target_ckpt):
    shutil.copy2(source, target_ckpt)
print(f"Downloaded checkpoint: {{target_ckpt}}")
PY
    status=$?
    rm -rf "$lock_dir"
    return "$status"
  fi

  echo "[draft-v100x4x8] waiting for checkpoint download lock: $lock_dir"
  for _ in $(seq 1 180); do
    if [[ -f "$CKPT_PATH" ]]; then
      echo "[draft-v100x4x8] checkpoint appeared: $CKPT_PATH"
      return 0
    fi
    sleep 10
  done
  echo "[draft-v100x4x8] timed out waiting for $CKPT_PATH" >&2
  return 4
}}

ensure_checkpoint() {{
  local metadata=""
  if (( NODE_RANK == 0 )); then
    ensure_checkpoint_local || return $?
    start_checkpoint_sync_server
    return $?
  fi
  wait_for_checkpoint_sync_server || return $?
  metadata="$(fetch_checkpoint_metadata)" || return $?
  download_synced_checkpoint "$metadata"
}}

trap stop_checkpoint_sync_server EXIT
ensure_checkpoint || exit $?

extra_overrides=()
if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
  read -r -a extra_overrides <<< "$CATK_EXTRA_OVERRIDES"
fi

torchrun_args=(
  --nnodes "$NNODES"
  --nproc_per_node "$NPROC_PER_NODE"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  -m src.run
  experiment="$EXPERIMENT"
  action=finetune
  trainer=ddp
  trainer.devices="$NPROC_PER_NODE"
  trainer.num_nodes="$NNODES"
  trainer.precision="$PRECISION"
  paths.cache_root="$CACHE_ROOT"
  paths.log_dir="$LOG_DIR"
  task_name="$TASK_NAME"
  ckpt_path="$CKPT_PATH"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.test_batch_size="$TEST_BATCH_SIZE"
  model.model_config.lr="$CATK_LR"
  model.model_config.draft.enabled=true
  model.model_config.draft.loss_enabled=true
  model.model_config.draft.physics.soft_limit_ratio=0.8
  model.model_config.draft.physics.topk_violation_k=20
  model.model_config.draft.physics.commit_loss_weight=1.0
  model.model_config.draft.physics.use_slip_penalty=false
)
torchrun_args+=("${{extra_overrides[@]}}")

printf '[draft-v100x4x8] torchrun'
printf ' %q' "${{torchrun_args[@]}}"
printf '\\n'

torchrun "${{torchrun_args[@]}}"
status=$?
exit "$status"
"""


def render_monitor_script(interval: int, task_name: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
while true; do
  echo
  echo "[monitor] $(date '+%F %T') task={task_name} pod=$(hostname)"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  sleep {int(interval)}
done
"""


def render_start_command(
    args: argparse.Namespace,
    *,
    pod: str,
    rank: int,
    master_addr: str,
) -> str:
    root = run_root(args)
    env_file = f"{root}/{pod}.env"
    worker_file = f"{root}/{pod}_worker.sh"
    monitor_file = f"{root}/{pod}_monitor.sh"
    tmux_log = f"{root}/{pod}.tmux.log"

    pull_block = ""
    if args.pull:
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(args.branch + ':refs/remotes/origin/' + args.branch)}
if git show-ref --verify --quiet {shq('refs/heads/' + args.branch)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq('origin/' + args.branch)}
fi
git pull --ff-only origin {shq(args.branch)}
"""

    replace_block = ""
    if args.replace:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  tmux kill-session -t {shq(args.session)}
fi
TASK_NAME_TO_STOP={shq(args.task_name)}
mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating stale task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing stale task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
"""
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
  echo "[launcher] attach with: tmux attach -t {args.session}" >&2
  exit 3
fi
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{render_monitor_script(args.monitor_interval, args.task_name).rstrip()}
CATK_MONITOR
chmod +x {shq(monitor_file)}
tmux split-window -v -l 12 -t {shq(args.session)} {shq(monitor_file)}
tmux select-pane -t {shq(args.session)}
"""

    return f"""set -Eeuo pipefail
if [ ! -d {shq(args.project_root)}/.git ]; then
  echo "[launcher] PROJECT_ROOT is not a git checkout: {args.project_root}" >&2
  exit 2
fi
cd {shq(args.project_root)}
{pull_block}
{replace_block}
mkdir -p {shq(root)}
cat > {shq(env_file)} <<'CATK_ENV'
{render_env(args, rank=rank, master_addr=master_addr).rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(env_file).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.session} on {pod}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_stop_command(session: str, task_name: str) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
fi
TASK_NAME_TO_STOP={shq(task_name)}
mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing task processes for $TASK_NAME_TO_STOP: ${{pids[*]}}"
    kill -KILL "${{pids[@]}}" 2>/dev/null || true
  fi
fi
"""


def exec_in_pod(args: argparse.Namespace, pod: str, script: str) -> None:
    command = [
        "exec",
        "-n",
        args.namespace,
        pod,
        "-c",
        args.container,
        "--",
        "bash",
        "-lc",
        script,
    ]
    if args.dry_run:
        print("kubectl " + " ".join(shq(part) for part in command))
        return
    run_kubectl(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch DRaFT fine-tuning on existing V100x4x8 static pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument(
        "--wandb-artifact",
        default="",
        help="Optional W&B artifact full name to download if --ckpt-path is missing.",
    )
    parser.add_argument(
        "--artifact-download-dir",
        default="",
        help="Optional artifact download directory. Defaults to dirname(--ckpt-path)/artifact.",
    )
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29593")
    parser.add_argument(
        "--checkpoint-sync-port",
        default="29594",
        help="Rank-0 pod HTTP port used to serve the resolved checkpoint to worker pods.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--test-batch-size", type=int, default=2)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--learning-rate", default="2.0e-4")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if len(args.pods) != 8:
        parser.error("--pods must contain exactly eight pods for the V100x4x8 preset")
    if args.nproc_per_node != 4:
        parser.error("--nproc-per-node must be 4 for the V100x4x8 preset")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be >= 1")
    if args.val_batch_size < 1:
        parser.error("--val-batch-size must be >= 1")
    if args.test_batch_size < 1:
        parser.error("--test-batch-size must be >= 1")
    if not args.ckpt_path:
        parser.error("--ckpt-path is required unless --stop is set")
    return args


def main() -> None:
    args = parse_args()

    if args.stop:
        for pod in args.pods:
            exec_in_pod(args, pod, render_stop_command(args.session, args.task_name))
        return

    master_addr = args.master_addr or (
        "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    )
    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] checkpoint sync: {master_addr}:{args.checkpoint_sync_port}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] ckpt path: {args.ckpt_path}")
    print(f"[launcher] batch:     train={args.train_batch_size} val={args.val_batch_size}")

    for rank, pod in enumerate(args.pods):
        script = render_start_command(args, pod=pod, rank=rank, master_addr=master_addr)
        exec_in_pod(args, pod, script)

    print("\nAttach commands:")
    for pod in args.pods:
        print(
            f"  kubectl exec -it -n {args.namespace} {pod} "
            f"-c {args.container} -- tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
