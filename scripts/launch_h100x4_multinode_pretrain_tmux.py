#!/usr/bin/env python3
"""Launch CAT-K H100x4 multi-node pretrain in tmux on existing pods.

This launcher never creates, deletes, or restarts pods. It only runs
``kubectl exec`` against already-running pods and starts/kills tmux sessions
inside them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = ["hsb-npc-training", "hsb-npc-training2"]
DEFAULT_BRANCH = os.environ.get("CATK_BRANCH", "self_forcing_bugfix")
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_CACHE_ROOT_BY_POD = {
    "hsb-npc-training": "/mnt/nuplan/womd_v1_3/SMART_cache",
    "hsb-npc-training2": "/workspace/womd_v1_3/SMART_cache",
}
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"


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
        [
            "get",
            "pod",
            pod,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.podIP}",
        ],
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
    count = int(output.strip())
    if count < 1:
        raise RuntimeError(f"no GPUs found in pod {pod}")
    return count


def export_line(name: str, value: object) -> str:
    return f"export {name}={shq(value)}"


def parse_pod_cache_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--pod-cache-root entries must use POD=PATH, "
                f"but got: {value!r}"
            )
        pod, path = value.split("=", 1)
        pod = pod.strip()
        path = path.strip()
        if not pod or not path:
            raise ValueError(
                "--pod-cache-root entries must include both POD and PATH, "
                f"but got: {value!r}"
            )
        roots[pod] = path
    return roots


def validate_nproc_per_node(value: str) -> str:
    if value in {"auto", "gpu"}:
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--nproc-per-node must be a positive integer or one of: auto, gpu"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--nproc-per-node must be >= 1")
    return value


def cache_root_for_pod(args: argparse.Namespace, pod: str) -> str:
    if pod in args.pod_cache_root_map:
        return args.pod_cache_root_map[pod]
    if args.cache_root:
        return args.cache_root
    return DEFAULT_CACHE_ROOT_BY_POD.get(pod, DEFAULT_CACHE_ROOT)


def render_env_file(
    *,
    args: argparse.Namespace,
    cache_root: str,
    rank: int,
    master_addr: str,
    task_name: str,
    run_root: str,
    local_world_size: int | None = None,
    manual_rank_offset: int | None = None,
    manual_world_size: int | None = None,
) -> str:
    nproc_per_node = local_world_size if local_world_size is not None else args.nproc_per_node
    lines = [
        export_line("CACHE_ROOT", cache_root),
        export_line("NNODES", len(args.pods)),
        export_line("NPROC_PER_NODE", nproc_per_node),
        export_line("NODE_RANK", rank),
        export_line("MASTER_ADDR", master_addr),
        export_line("MASTER_PORT", args.master_port),
        export_line("CHECKPOINT_SYNC_HOST", master_addr),
        export_line("CHECKPOINT_SYNC_PORT", args.checkpoint_sync_port),
        export_line("TASK_NAME", task_name),
        export_line("CATK_EXPERIMENT", args.experiment),
        export_line("CATK_ACTION", args.action),
        export_line("LOG_DIR", args.log_dir),
        export_line("RUN_ROOT", run_root),
    ]
    if manual_rank_offset is not None and manual_world_size is not None:
        lines.extend(
            [
                export_line("MANUAL_RANK_OFFSET", manual_rank_offset),
                export_line("MANUAL_WORLD_SIZE", manual_world_size),
                export_line("TRAINER_DEVICES", nproc_per_node),
            ]
        )
    optional_env = {
        "CATK_CKPT_PATH": args.ckpt_path,
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "ACCUMULATE_GRAD_BATCHES": args.accumulate_grad_batches,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "CATK_LR": args.learning_rate,
        "CATK_HYDRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional_env.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
    return "\n".join(lines) + "\n"


def render_run_script(project_root: str, env_file: str) -> str:
    return f"""#!/usr/bin/env bash
set +e
export TERM="${{TERM:-xterm-256color}}"
export PYTHONUNBUFFERED=1
export CATK_REMOTE_PYTHON="${{CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}}"

cd {shq(project_root)}
set -a
source {shq(env_file)}
set +a

echo "[tmux-run] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[tmux-run] started at $(date '+%F %T')"
echo "[tmux-run] attach survives after exit; press Ctrl-b d to detach"
echo

mkdir -p "$RUN_ROOT"
torch_status_file="$RUN_ROOT/$(hostname).torchrun_status"
torch_pgid_file="$RUN_ROOT/$(hostname).torchrun_pgid"
CHECKPOINT_SYNC_PID=""
rm -f "$torch_status_file" "$torch_pgid_file"

start_checkpoint_sync_server() {{
  if (( NODE_RANK != 0 )); then
    return 0
  fi
  if [[ -z "${{CATK_CKPT_PATH:-}}" ]]; then
    return 0
  fi
  "$CATK_REMOTE_PYTHON" - <<'PY' &
import hashlib
import http.server
import os
import pathlib

ckpt_path = pathlib.Path(os.environ["CATK_CKPT_PATH"])
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
  echo "[tmux-run] checkpoint sync server started on $CHECKPOINT_SYNC_HOST:$CHECKPOINT_SYNC_PORT pid=$CHECKPOINT_SYNC_PID"
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
    if "$CATK_REMOTE_PYTHON" - <<'PY' >/dev/null 2>&1
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
      echo "[tmux-run] timed out waiting for checkpoint sync server at $CHECKPOINT_SYNC_HOST:$CHECKPOINT_SYNC_PORT" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
}}

fetch_checkpoint_metadata() {{
  "$CATK_REMOTE_PYTHON" - <<'PY'
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

sync_checkpoint_if_needed() {{
  local metadata=""
  local expected_size=""
  local expected_sha=""
  local tmp_file=""

  if [[ -z "${{CATK_CKPT_PATH:-}}" ]]; then
    return 0
  fi
  if (( NODE_RANK == 0 )); then
    if [[ ! -f "$CATK_CKPT_PATH" ]]; then
      echo "[tmux-run] ERROR: rank0 checkpoint does not exist: $CATK_CKPT_PATH" >&2
      return 1
    fi
    start_checkpoint_sync_server
    return $?
  fi

  wait_for_checkpoint_sync_server || return $?
  metadata="$(fetch_checkpoint_metadata)" || return $?
  expected_size="$(metadata_value "$metadata" size)"
  expected_sha="$(metadata_value "$metadata" sha256)"
  if [[ -z "$expected_size" || -z "$expected_sha" ]]; then
    echo "[tmux-run] invalid checkpoint metadata from rank0" >&2
    return 1
  fi
  if checkpoint_matches "$CATK_CKPT_PATH" "$expected_size" "$expected_sha"; then
    echo "[tmux-run] checkpoint already synced: $CATK_CKPT_PATH"
    return 0
  fi

  mkdir -p "$(dirname "$CATK_CKPT_PATH")"
  tmp_file="${{CATK_CKPT_PATH}}.download.$$"
  rm -f "$tmp_file"
  echo "[tmux-run] downloading rank0 checkpoint to $CATK_CKPT_PATH"
  if ! "$CATK_REMOTE_PYTHON" - "$tmp_file" <<'PY'
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
    echo "[tmux-run] downloaded checkpoint failed verification: $tmp_file" >&2
    rm -f "$tmp_file"
    return 1
  fi
  mv -f "$tmp_file" "$CATK_CKPT_PATH"
  checkpoint_matches "$CATK_CKPT_PATH" "$expected_size" "$expected_sha"
}}

terminate_process_group() {{
  local pgid="$1"
  if [[ -z "$pgid" || "$pgid" == "0" ]]; then
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep "${{REMOTE_KILL_GRACE_SEC:-20}}"
  kill -KILL -- "-$pgid" 2>/dev/null || true
}}

task_process_pids() {{
  pgrep -f "task_name=${{TASK_NAME}}" 2>/dev/null | while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" && "$pid" != "${{BASHPID:-}}" ]]; then
      echo "$pid"
    fi
  done
}}

terminate_task_processes() {{
  local reason="${{1:-cleanup}}"
  local grace_sec="${{TASK_PROCESS_KILL_GRACE_SEC:-15}}"
  local waited=0
  local pids=()

  mapfile -t pids < <(task_process_pids || true)
  if (( ${{#pids[@]}} == 0 )); then
    return 0
  fi

  echo "[tmux-run] terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  while (( waited < grace_sec )); do
    sleep 1
    waited=$(( waited + 1 ))
    mapfile -t pids < <(task_process_pids || true)
    if (( ${{#pids[@]}} == 0 )); then
      return 0
    fi
  done

  echo "[tmux-run] force killing task processes for $reason: ${{pids[*]}}"
  kill -KILL "${{pids[@]}}" 2>/dev/null || true
}}

terminate_attempt_processes() {{
  local pgid=""
  pgid="$(cat "$torch_pgid_file" 2>/dev/null || true)"
  terminate_process_group "$pgid"
  terminate_task_processes "$1"
}}

trap stop_checkpoint_sync_server EXIT
sync_checkpoint_if_needed || exit $?
terminate_task_processes "pre-run stale cleanup"

(
  set +e
  setsid bash -c 'pgid_file="$1"; shift; echo "$$" > "$pgid_file"; exec "$@"' \
    bash "$torch_pgid_file" bash scripts/h100x4_multinode_pretrain.sh
  echo "$?" > "$torch_status_file"
) &
runner_pid=$!

wait "$runner_pid"
status="$(cat "$torch_status_file" 2>/dev/null || echo 1)"
if [[ "$status" != "0" ]]; then
  terminate_attempt_processes "post-run cleanup status=$status"
fi

echo
echo "[tmux-run] exited with status $status at $(date '+%F %T')"
echo "[tmux-run] leaving shell open for inspection"
exec bash
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
    *,
    args: argparse.Namespace,
    pod: str,
    rank: int,
    master_addr: str,
    task_name: str,
    local_world_size: int | None = None,
    manual_rank_offset: int | None = None,
    manual_world_size: int | None = None,
) -> str:
    safe_task = task_name.replace("/", "_")
    run_root = f"{args.log_dir.rstrip('/')}/tmux_h100x4_multinode_pretrain/{safe_task}"
    env_file = f"{run_root}/{pod}.env"
    run_file = f"{run_root}/{pod}_run.sh"
    monitor_file = f"{run_root}/{pod}_monitor.sh"
    log_file = f"{run_root}/{pod}.tmux.log"
    pipe_command = f"cat >> {shq(log_file)}"
    cache_root = cache_root_for_pod(args, pod)
    env_text = render_env_file(
        args=args,
        cache_root=cache_root,
        rank=rank,
        master_addr=master_addr,
        task_name=task_name,
        run_root=run_root,
        local_world_size=local_world_size,
        manual_rank_offset=manual_rank_offset,
        manual_world_size=manual_world_size,
    )
    run_text = render_run_script(args.project_root, env_file)
    monitor_text = render_monitor_script(args.monitor_interval, task_name)

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
  echo "[launcher] attach with: tmux attach -t {args.session}" >&2
  exit 3
fi
"""

    pull_block = ""
    if args.pull:
        branch_ref = f"refs/heads/{args.branch}"
        origin_ref = f"origin/{args.branch}"
        fetch_refspec = f"{args.branch}:refs/remotes/origin/{args.branch}"
        pull_block = f"""
git config --global --add safe.directory {shq(args.project_root)} || true
git fetch origin {shq(fetch_refspec)}
if git show-ref --verify --quiet {shq(branch_ref)}; then
  git checkout {shq(args.branch)}
else
  git checkout -b {shq(args.branch)} {shq(origin_ref)}
fi
git pull --ff-only origin {shq(args.branch)}
"""
        if args.git_ref:
            pull_block += f"""
git fetch origin {shq(args.branch)} --tags || true
git checkout --detach {shq(args.git_ref)}
"""

    monitor_block = ""
    if not args.no_monitor_pane:
        monitor_block = f"""
cat > {shq(monitor_file)} <<'CATK_MONITOR'
{monitor_text.rstrip()}
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
mkdir -p {shq(run_root)}
cat > {shq(env_file)} <<'CATK_ENV'
{env_text.rstrip()}
CATK_ENV
cat > {shq(run_file)} <<'CATK_RUN'
{run_text.rstrip()}
CATK_RUN
chmod +x {shq(run_file)}
: > {shq(log_file)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq(pipe_command)}
{monitor_block}
echo "[launcher] started tmux session {args.session} on pod {pod}"
echo "[launcher] cache root: {cache_root}"
echo "[launcher] tmux log: {log_file}"
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


def exec_in_pod(
    namespace: str,
    container: str,
    pod: str,
    script: str,
    *,
    dry_run: bool,
) -> None:
    cmd = [
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
    ]
    if dry_run:
        print("kubectl " + " ".join(shq(part) for part in cmd))
        return
    run_kubectl(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start CAT-K H100x4 multi-node pretrain in tmux on existing pods.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pods", nargs="+", default=DEFAULT_PODS)
    parser.add_argument("--container", default="main")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--git-ref",
        default="",
        help="Optional exact commit/tag to check out after updating --branch on each pod.",
    )
    parser.add_argument("--no-pull", dest="pull", action="store_false")
    parser.set_defaults(pull=True)
    parser.add_argument(
        "--cache-root",
        default="",
        help=(
            "Use one CACHE_ROOT for every pod. If omitted, known pod-specific "
            "defaults are used."
        ),
    )
    parser.add_argument(
        "--pod-cache-root",
        action="append",
        default=[],
        metavar="POD=PATH",
        help=(
            "Override CACHE_ROOT for one pod. Can be repeated; takes priority "
            "over --cache-root."
        ),
    )
    parser.add_argument(
        "--action",
        choices=["fit", "validate", "test"],
        default="fit",
    )
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--experiment", default="pre_bc_flow_2x4_h100")
    parser.add_argument("--task-name", default="")
    parser.add_argument("--session", default="catk-h100x4-pretrain")
    parser.add_argument("--master-addr", default="")
    parser.add_argument("--master-port", default="29511")
    parser.add_argument(
        "--checkpoint-sync-port",
        default="29512",
        help="Rank-0 pod HTTP port used to serve --ckpt-path to worker pods.",
    )
    parser.add_argument("--nproc-per-node", type=validate_nproc_per_node, default="4")
    parser.add_argument(
        "--manual-rank-offsets",
        action="store_true",
        help=(
            "Launch one process per local GPU with explicit RANK/WORLD_SIZE "
            "env vars instead of torchrun. Use for heterogeneous pod GPU counts."
        ),
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--train-batch-size", default="")
    parser.add_argument("--val-batch-size", default="")
    parser.add_argument("--accumulate-grad-batches", default="")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--learning-rate", default="")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        args.pod_cache_root_map = parse_pod_cache_roots(args.pod_cache_root)
    except ValueError as exc:
        parser.error(str(exc))

    if len(args.pods) < 2 and not args.stop:
        parser.error("--pods must contain at least two pods for multi-node training")
    if args.manual_rank_offsets and args.action != "fit" and not args.stop:
        parser.error("--manual-rank-offsets is only wired for action=fit")
    if args.monitor_interval < 1:
        parser.error("--monitor-interval must be >= 1")
    if args.action in {"validate", "test"} and not args.ckpt_path and not args.stop:
        parser.error(f"--ckpt-path is required when --action={args.action}")
    if not args.task_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.task_name = f"flow_semi_continuous_pretrain_h100x4x{len(args.pods)}_{stamp}"
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        for pod in args.pods:
            exec_in_pod(
                args.namespace,
                args.container,
                pod,
                render_stop_command(args.session, args.task_name),
                dry_run=args.dry_run,
            )
        return

    master_addr = args.master_addr or (
        "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    )
    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] checkpoint sync: {master_addr}:{args.checkpoint_sync_port}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print("[launcher] cache roots:")
    for pod in args.pods:
        print(f"  {pod}: {cache_root_for_pod(args, pod)}")

    local_world_sizes: dict[str, int] = {}
    rank_offsets: dict[str, int] = {}
    manual_world_size: int | None = None
    if args.manual_rank_offsets:
        offset = 0
        for pod in args.pods:
            if args.dry_run and args.nproc_per_node in {"auto", "gpu"}:
                local_size = 1
            elif args.nproc_per_node in {"auto", "gpu"}:
                local_size = pod_gpu_count(args.namespace, args.container, pod)
            else:
                local_size = int(args.nproc_per_node)
            local_world_sizes[pod] = local_size
            rank_offsets[pod] = offset
            offset += local_size
        manual_world_size = offset
        print("[launcher] manual rank layout:")
        for pod in args.pods:
            print(
                f"  {pod}: local_world_size={local_world_sizes[pod]} "
                f"rank_offset={rank_offsets[pod]}"
            )
        print(f"[launcher] manual world_size: {manual_world_size}")

    for rank, pod in enumerate(args.pods):
        script = render_start_command(
            args=args,
            pod=pod,
            rank=rank,
            master_addr=master_addr,
            task_name=args.task_name,
            local_world_size=local_world_sizes.get(pod),
            manual_rank_offset=rank_offsets.get(pod),
            manual_world_size=manual_world_size,
        )
        exec_in_pod(args.namespace, args.container, pod, script, dry_run=args.dry_run)

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
