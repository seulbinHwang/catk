#!/usr/bin/env python3
"""Launch self-forced V100x4x4 static multi-node training on existing pods."""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["testsv", "testsvv", "testsvvv", "testsvvvv"]
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_v100x4x4"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-sjan8kmh:v32"
)
DEFAULT_PRETRAIN_CKPT = (
    "/mnt/nuplan/projects/catk/downloads/wandb_ckpts/"
    "flow_semi_continuous_finetune_inv_euler_32_a100x4/"
    "epoch-last-sjan8kmh_v32/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/mnt/nuplan/projects/catk/downloads/wandb_ckpts/"
    "flow_semi_continuous_finetune_inv_euler_32_a100x4/"
    "epoch-last-sjan8kmh_v32/artifact"
)
DEFAULT_TASK_NAME = (
    "flow_self_forced_v100x4x4_"
    "unfrozen_except_map_encoder_estimator_warmup_1_bs4"
)
DEFAULT_SESSION = "catk-sf-v100x4x4-exceptmap-warmup1"


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


def render_env(args: argparse.Namespace, *, rank: int, master_addr: str) -> str:
    lines = [
        export_line("PROJECT_ROOT", args.project_root),
        export_line("CACHE_ROOT", args.cache_root),
        export_line("PRETRAIN_CKPT", args.pretrain_ckpt),
        export_line("WANDB_PRETRAIN_ARTIFACT", args.wandb_pretrain_artifact),
        export_line("WANDB_PRETRAIN_DOWNLOAD_DIR", args.pretrain_download_dir),
        export_line("EXPERIMENT", args.experiment),
        export_line("TASK_NAME", args.task_name),
        export_line("NNODES", len(args.pods)),
        export_line("NPROC_PER_NODE", args.nproc_per_node),
        export_line("NODE_RANK", rank),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("RETRY_SYNC_HOST", master_addr),
        export_line("RETRY_SYNC_PORT", args.retry_sync_port),
        export_line("INITIAL_BS", args.initial_bs),
        export_line("OOM_STEP", args.oom_step),
        export_line("MIN_BS", args.min_bs),
        export_line("VAL_BATCH_SIZE", args.val_batch_size),
        export_line("TEST_BATCH_SIZE", args.test_batch_size),
        export_line("PRECISION", args.precision),
        export_line("SCORER_SCENE_NUM", args.scorer_scene_num),
        export_line("UNFROZEN_RANGE", args.unfrozen_range),
        export_line("ESTIMATOR_WARMUP_EPOCHS", args.estimator_warmup_epochs),
        export_line("LOG_DIR", args.log_dir),
        export_line("RUN_ROOT", run_root(args)),
        export_line("RETRY_STATE_DIR", f"{run_root(args)}/retry_state"),
    ]
    optional = {
        "CATK_LR": args.learning_rate,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "TRAIN_EPOCH_SAMPLE_FRACTION": args.train_epoch_sample_fraction,
        "CATK_EXTRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
    return "\n".join(lines) + "\n"


def run_root(args: argparse.Namespace) -> str:
    safe_task = args.task_name.replace("/", "_")
    return f"{args.log_dir.rstrip('/')}/tmux_static_multinode/{safe_task}"


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
mkdir -p "$RUN_ROOT" "$RETRY_STATE_DIR"

echo "[self-forced-v100x4x4] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[self-forced-v100x4x4] started at $(date '+%F %T')"
echo "[self-forced-v100x4x4] experiment=${{EXPERIMENT}} bs=${{INITIAL_BS}} precision=${{PRECISION}}"
echo "[self-forced-v100x4x4] pretrain_artifact=${{WANDB_PRETRAIN_ARTIFACT}}"
echo "[self-forced-v100x4x4] pretrain_ckpt=${{PRETRAIN_CKPT}}"
echo "[self-forced-v100x4x4] attach survives after exit; press Ctrl-b d to detach"
echo

OOM_REGEX='OutOfMemoryError|CUDA out of memory|c10::OutOfMemoryError|torch\\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY'
RETRY_SYNC_PID=""

start_retry_sync_server() {{
  if (( NODE_RANK != 0 )); then
    return 0
  fi

  python - <<'PY' &
import http.server
import os
import pathlib
import re
import urllib.parse

root = pathlib.Path(os.environ["RETRY_STATE_DIR"])
root.mkdir(parents=True, exist_ok=True)
port = int(os.environ["RETRY_SYNC_PORT"])
name_re = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_statuses(attempt):
    files = sorted(root.glob(attempt + ".*.status"))
    status_values = []
    oom_values = []
    for path in files:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        values = dict(line.split("=", 1) for line in data if "=" in line)
        status_values.append(values.get("status", "1"))
        oom_values.append(values.get("oom", "0"))
    failure = any(value != "0" for value in status_values)
    oom = any(value == "1" for value in oom_values)
    return len(files), failure, oom


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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.strip("/")
        if path == "health":
            self.send_text(200, "ok\\n")
            return
        parts = path.split("/", 1)
        if len(parts) != 2 or parts[0] not in ("count", "aggregate"):
            self.send_text(404, "not found\\n")
            return
        attempt = parts[1]
        if not name_re.match(attempt):
            self.send_text(400, "bad attempt\\n")
            return
        count, failure, oom = read_statuses(attempt)
        if parts[0] == "count":
            self.send_text(200, str(count) + "\\n")
        else:
            self.send_text(
                200,
                "count=" + str(count) + "\\n"
                + "failure=" + ("1" if failure else "0") + "\\n"
                + "oom=" + ("1" if oom else "0") + "\\n",
            )

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/status":
            self.send_text(404, "not found\\n")
            return
        query = urllib.parse.parse_qs(parsed.query)
        attempt = query.get("attempt", [""])[0]
        host = query.get("host", [""])[0]
        if not name_re.match(attempt) or not name_re.match(host):
            self.send_text(400, "bad name\\n")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        target = root / (attempt + "." + host + ".status")
        tmp = root / (target.name + ".tmp." + str(os.getpid()))
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(target)
        self.send_text(200, "ok\\n")


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


Server(("", port), Handler).serve_forever()
PY
  RETRY_SYNC_PID=$!
  echo "$RETRY_SYNC_PID" > "$RETRY_STATE_DIR/sync_server.pid"
  echo "[self-forced-v100x4x4] retry sync server started on $MASTER_ADDR:$RETRY_SYNC_PORT pid=$RETRY_SYNC_PID"
}}

stop_retry_sync_server() {{
  if [[ -n "$RETRY_SYNC_PID" ]]; then
    kill "$RETRY_SYNC_PID" 2>/dev/null || true
  fi
}}

retry_sync_get() {{
  local endpoint="$1"
  python - "$endpoint" <<'PY'
import os
import sys
import urllib.request

endpoint = sys.argv[1].lstrip("/")
url = "http://" + os.environ["RETRY_SYNC_HOST"] + ":" + os.environ["RETRY_SYNC_PORT"] + "/" + endpoint
with urllib.request.urlopen(url, timeout=5) as response:
    sys.stdout.write(response.read().decode("utf-8"))
PY
}}

wait_for_retry_sync_server() {{
  local waited=0
  local timeout_sec="${{RETRY_SYNC_START_TIMEOUT_SEC:-120}}"
  while true; do
    if retry_sync_get "health" >/dev/null 2>&1; then
      return 0
    fi
    if (( waited >= timeout_sec )); then
      echo "[self-forced-v100x4x4] retry sync server did not become ready at $RETRY_SYNC_HOST:$RETRY_SYNC_PORT" >&2
      return 1
    fi
    sleep 2
    waited=$(( waited + 2 ))
  done
}}

post_attempt_status() {{
  local attempt_name="$1"
  local status_file="$2"
  local try
  for try in $(seq 1 12); do
    if python - "$attempt_name" "$status_file" <<'PY'
import os
import socket
import sys
import urllib.parse
import urllib.request

attempt = sys.argv[1]
status_file = sys.argv[2]
host = socket.gethostname()
query = urllib.parse.urlencode((("attempt", attempt), ("host", host)))
url = "http://" + os.environ["RETRY_SYNC_HOST"] + ":" + os.environ["RETRY_SYNC_PORT"] + "/status?" + query
with open(status_file, "rb") as handle:
    body = handle.read()
request = urllib.request.Request(url, data=body, method="POST")
with urllib.request.urlopen(request, timeout=10) as response:
    response.read()
PY
    then
      return 0
    fi
    sleep 5
  done
  echo "[self-forced-v100x4x4] failed to post retry status for $attempt_name after retries" >&2
  return 1
}}

ensure_pretrain_checkpoint() {{
  if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[self-forced-v100x4x4] using cached pretrain checkpoint: $PRETRAIN_CKPT"
    return 0
  fi

  mkdir -p "$(dirname "$PRETRAIN_CKPT")" "$WANDB_PRETRAIN_DOWNLOAD_DIR"
  lock_dir="${{PRETRAIN_CKPT}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[self-forced-v100x4x4] downloading W&B artifact: $WANDB_PRETRAIN_ARTIFACT"
    python - <<'PY'
import glob
import os
import shutil
import sys
from pathlib import Path

artifact_name = os.environ["WANDB_PRETRAIN_ARTIFACT"]
download_dir = os.environ["WANDB_PRETRAIN_DOWNLOAD_DIR"]
target_ckpt = os.environ["PRETRAIN_CKPT"]

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
print(f"Downloaded pretrain checkpoint: {{target_ckpt}}")
PY
    status=$?
    rm -rf "$lock_dir"
    if (( status != 0 )); then
      echo "[self-forced-v100x4x4] W&B artifact download failed with status $status" >&2
      return "$status"
    fi
  else
    echo "[self-forced-v100x4x4] waiting for checkpoint download lock: $lock_dir"
    for _ in $(seq 1 180); do
      if [[ -f "$PRETRAIN_CKPT" ]]; then
        echo "[self-forced-v100x4x4] checkpoint appeared: $PRETRAIN_CKPT"
        return 0
      fi
      sleep 10
    done
    echo "[self-forced-v100x4x4] timed out waiting for $PRETRAIN_CKPT" >&2
    return 4
  fi

  test -f "$PRETRAIN_CKPT"
}}

find_latest_self_forced_ckpt() {{
  ls -t "$PROJECT_ROOT/logs/$TASK_NAME/runs"/*/checkpoints/epoch_last.ckpt 2>/dev/null | head -1
}}

write_attempt_status() {{
  local status="$1"
  local oom="$2"
  local status_file="$RETRY_STATE_DIR/$attempt_tag.$(hostname).status"
  local tmp_file="$status_file.$$"
  {{
    echo "host=$(hostname)"
    echo "node_rank=$NODE_RANK"
    echo "attempt=$attempt"
    echo "batch_size=$bs"
    echo "status=$status"
    echo "oom=$oom"
    echo "log=$attempt_log"
    echo "time=$(date '+%F %T')"
  }} > "$tmp_file"
  mv "$tmp_file" "$status_file"
  post_attempt_status "$attempt_tag" "$status_file"
}}

wait_for_attempt_statuses() {{
  local waited=0
  local timeout_sec="${{RETRY_BARRIER_TIMEOUT_SEC:-1200}}"
  local count=0
  while true; do
    count="$(retry_sync_get "count/$attempt_tag" 2>/dev/null | tr -d '[:space:]')"
    count="${{count:-0}}"
    if (( count >= NNODES )); then
      return 0
    fi
    if (( waited >= timeout_sec )); then
      echo "[self-forced-v100x4x4] timed out waiting for attempt $attempt status files: got $count/$NNODES" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
}}

global_attempt_has_oom() {{
  retry_sync_get "aggregate/$attempt_tag" 2>/dev/null | grep -q '^oom=1$'
}}

global_attempt_has_failure() {{
  retry_sync_get "aggregate/$attempt_tag" 2>/dev/null | grep -q '^failure=1$'
}}

start_retry_sync_server
trap stop_retry_sync_server EXIT
wait_for_retry_sync_server || exit 1
ensure_pretrain_checkpoint || exit $?

bs="$INITIAL_BS"
attempt=0
while (( bs >= MIN_BS )); do
  attempt=$(( attempt + 1 ))
  attempt_tag="attempt_$(printf '%03d' "$attempt")_bs${{bs}}"
  attempt_log="$RUN_ROOT/$(hostname).${{attempt_tag}}.log"

  latest_ckpt="$(find_latest_self_forced_ckpt)"
  if [[ -n "$latest_ckpt" ]]; then
    action="fit"
    ckpt_path="$latest_ckpt"
  else
    action="finetune"
    ckpt_path="$PRETRAIN_CKPT"
  fi

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
    action="$action"
    paths.cache_root="$CACHE_ROOT"
    paths.log_dir="$LOG_DIR"
    task_name="$TASK_NAME"
    ckpt_path="$ckpt_path"
    trainer.devices="$NPROC_PER_NODE"
    trainer.num_nodes="$NNODES"
    trainer.precision="$PRECISION"
    data.train_batch_size="$bs"
    data.val_batch_size="$VAL_BATCH_SIZE"
    data.test_batch_size="$TEST_BATCH_SIZE"
    model.model_config.scorer_scene_num="$SCORER_SCENE_NUM"
    model.model_config.self_forced.unfrozen_range="$UNFROZEN_RANGE"
    model.model_config.self_forced.estimator_warmup_epochs="$ESTIMATOR_WARMUP_EPOCHS"
  )

  if [[ -n "${{CATK_LR:-}}" ]]; then
    torchrun_args+=(model.model_config.lr="$CATK_LR")
  fi
  if [[ -n "${{LIMIT_TRAIN_BATCHES:-}}" ]]; then
    torchrun_args+=(trainer.limit_train_batches="$LIMIT_TRAIN_BATCHES")
  fi
  if [[ -n "${{LIMIT_VAL_BATCHES:-}}" ]]; then
    torchrun_args+=(trainer.limit_val_batches="$LIMIT_VAL_BATCHES")
  fi
  if [[ -n "${{MAX_EPOCHS:-}}" ]]; then
    torchrun_args+=(trainer.max_epochs="$MAX_EPOCHS")
  fi
  if [[ -n "${{TRAIN_EPOCH_SAMPLE_FRACTION:-}}" ]]; then
    torchrun_args+=(data.train_epoch_sample_fraction="$TRAIN_EPOCH_SAMPLE_FRACTION")
  fi
  torchrun_args+=("${{extra_overrides[@]}}")

  echo
  echo "[self-forced-v100x4x4] $attempt_tag action=$action ckpt=$ckpt_path"
  printf '[self-forced-v100x4x4] torchrun'
  printf ' %q' "${{torchrun_args[@]}}"
  printf '\\n'

  torchrun "${{torchrun_args[@]}}" 2>&1 | tee "$attempt_log"
  status=${{PIPESTATUS[0]}}
  local_oom=0
  if grep -Eq "$OOM_REGEX" "$attempt_log" 2>/dev/null; then
    local_oom=1
  fi
  write_attempt_status "$status" "$local_oom"

  if ! wait_for_attempt_statuses; then
    echo "[self-forced-v100x4x4] retry barrier failed for $attempt_tag; see $RETRY_STATE_DIR" >&2
    exit 1
  fi

  if ! global_attempt_has_failure; then
    echo "[self-forced-v100x4x4] training completed successfully at bs=$bs"
    exit 0
  fi

  if global_attempt_has_oom; then
    next_bs=$(( bs - OOM_STEP ))
    echo "[self-forced-v100x4x4] OOM detected on at least one node in attempt $attempt; all nodes lowering bs $bs -> $next_bs"
    bs="$next_bs"
    continue
  fi

  echo "[self-forced-v100x4x4] non-OOM failure status=$status; see $attempt_log and $RETRY_STATE_DIR/$attempt_tag.*.status"
  exit "$status"
done

echo "[self-forced-v100x4x4] reached MIN_BS=$MIN_BS without success"
exit 1
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

    if rank == 0:
        state_reset_block = f"""
rm -rf {shq(root + '/retry_state')}
mkdir -p {shq(root + '/retry_state')}
"""
    else:
        state_reset_block = f"""
mkdir -p {shq(root + '/retry_state')}
"""

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
"""
    else:
        replace_block = f"""
if tmux has-session -t {shq(args.session)} 2>/dev/null; then
  echo "[launcher] tmux session already exists: {args.session}" >&2
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
tmux select-pane -t {shq(args.session)}:0.0
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
{state_reset_block}
cat > {shq(env_file)} <<'CATK_ENV'
{render_env(args, rank=rank, master_addr=master_addr).rstrip()}
CATK_ENV
cat > {shq(worker_file)} <<'CATK_WORKER'
{render_worker_script(env_file).rstrip()}
CATK_WORKER
chmod +x {shq(worker_file)}
: > {shq(tmux_log)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(worker_file)}
tmux pipe-pane -t {shq(args.session)}:0.0 -o {shq('cat >> ' + shq(tmux_log))}
{monitor_block}
echo "[launcher] started {args.session} on {pod}"
echo "[launcher] tmux log: {tmux_log}"
"""


def render_stop_command(session: str) -> str:
    return f"""set -Eeuo pipefail
if tmux has-session -t {shq(session)} 2>/dev/null; then
  tmux kill-session -t {shq(session)}
  echo "[launcher] stopped tmux session {session}"
else
  echo "[launcher] tmux session not found: {session}"
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
        description="Launch self-forced V100x4x4 training on existing static pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--pretrain-ckpt",
        default=DEFAULT_PRETRAIN_CKPT,
        help=(
            "Local checkpoint path. If missing, the launcher downloads "
            "WANDB_PRETRAIN_ARTIFACT here before training."
        ),
    )
    parser.add_argument("--wandb-pretrain-artifact", default=DEFAULT_WANDB_PRETRAIN_ARTIFACT)
    parser.add_argument("--pretrain-download-dir", default=DEFAULT_PRETRAIN_DOWNLOAD_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29543")
    parser.add_argument(
        "--retry-sync-port",
        default="29544",
        help="Rank-0 pod HTTP port used to collect retry status from all pods.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--initial-bs", type=int, default=4)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--scorer-scene-num", type=int, default=320)
    parser.add_argument("--unfrozen-range", default="except_map_encoder")
    parser.add_argument("--estimator-warmup-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--train-epoch-sample-fraction", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if len(args.pods) < 2:
        parser.error("--pods must contain at least two pods")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if not args.pretrain_ckpt:
        parser.error("--pretrain-ckpt must not be empty unless --stop is set")
    if not args.wandb_pretrain_artifact:
        parser.error("--wandb-pretrain-artifact must not be empty unless --stop is set")
    if not args.pretrain_download_dir:
        parser.error("--pretrain-download-dir must not be empty unless --stop is set")
    return args


def main() -> None:
    args = parse_args()

    if args.stop:
        for pod in args.pods:
            exec_in_pod(args, pod, render_stop_command(args.session))
        return

    master_addr = args.master_addr or pod_ip(args.namespace, args.pods[0])
    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] retry sync: {master_addr}:{args.retry_sync_port}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] bs fallback: {args.initial_bs}->{args.min_bs} step {args.oom_step}")

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
