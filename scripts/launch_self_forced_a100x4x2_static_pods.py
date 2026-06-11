#!/usr/bin/env python3
"""Launch self-forced A100x4x2 static multi-node training on existing pods.

This launcher never creates, deletes, or restarts pods. It only uses
``kubectl exec`` to start or stop a tmux session inside already-running pods.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_PODS = [
    "testa",
    "testaa",
]
DEFAULT_CONTAINER = "main"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "self_forcing_w_track_loss"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_npfm_a100x4x2"
DEFAULT_WANDB_PRETRAIN_ARTIFACT = (
    "jksg01019-naver-labs/SMART-FLOW/epoch-last-g3zr84tp:v64"
)
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "v64/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_semi_continuous_pretrain_h100x4x2_bs26/"
    "v64/artifact"
)
DEFAULT_TASK_NAME = (
    "flow_self_forced_a100x4x2_"
    "use_stop_motion_false_estimator_warmup_1_lr1e-6_bs22"
)
DEFAULT_SESSION = "catk-sf-a100x4x2-stopfalse-warmup1"
DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT = (
    "generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6:latest"
)
DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME = (
    "generated-estimator-warmup-bank-pretrain-x5f9g0ce-v57-lr1e-6"
)


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
        export_line("ESTIMATOR_WARMUP_BANK_ENABLED", str(args.use_estimator_warmup_bank).lower()),
        export_line("ESTIMATOR_WARMUP_BANK_ARTIFACT", args.estimator_warmup_bank_artifact),
        export_line("ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME", args.estimator_warmup_bank_artifact_name),
        export_line("ESTIMATOR_WARMUP_BANK_ENTITY", args.estimator_warmup_bank_entity),
        export_line("ESTIMATOR_WARMUP_BANK_PROJECT", args.estimator_warmup_bank_project),
        export_line("ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS", str(args.estimator_warmup_bank_adjust_max_epochs).lower()),
        export_line("SELF_FORCED_USE_STOP_MOTION", args.self_forced_use_stop_motion),
        export_line("LOG_DIR", args.log_dir),
        export_line("RUN_ROOT", run_root(args)),
        export_line("RETRY_STATE_DIR", f"{run_root(args)}/retry_state"),
        export_line("INITIAL_ACTION", args.initial_action),
    ]
    optional = {
        "CATK_LR": args.learning_rate,
        "CATK_GENERATED_ESTIMATOR_LR": args.generated_estimator_learning_rate,
        "CATK_LR_COSINE_FINAL_RATIO": args.lr_cosine_final_ratio,
        "DECODER_USE_STOP_MOTION": args.decoder_use_stop_motion,
        "LIMIT_TRAIN_BATCHES": args.limit_train_batches,
        "LIMIT_VAL_BATCHES": args.limit_val_batches,
        "MAX_EPOCHS": args.max_epochs,
        "CHECK_VAL_EVERY_N_EPOCH": args.check_val_every_n_epoch,
        "TRAIN_EPOCH_SAMPLE_FRACTION": args.train_epoch_sample_fraction,
        "TRAIN_MEMORY_BALANCED_BATCHES": args.train_memory_balanced_batches,
        "CATK_EXTRA_OVERRIDES": args.extra_hydra_overrides,
    }
    for name, value in optional.items():
        if value not in (None, ""):
            lines.append(export_line(name, value))
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
mkdir -p "$RUN_ROOT" "$RETRY_STATE_DIR"

ESTIMATOR_WARMUP_BANK_INIT_PATH=""
ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH=""
ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP="${{ESTIMATOR_WARMUP_EPOCHS:-}}"
ESTIMATOR_WARMUP_BANK_LOADED_WARMUP=0
ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="${{ESTIMATOR_WARMUP_EPOCHS:-0}}"
ESTIMATOR_WARMUP_BANK_LR=""

echo "[self-forced-a100x4x2] pod=$(hostname) rank=${{NODE_RANK}} task=${{TASK_NAME}}"
echo "[self-forced-a100x4x2] started at $(date '+%F %T')"
echo "[self-forced-a100x4x2] experiment=${{EXPERIMENT}} bs=${{INITIAL_BS}} precision=${{PRECISION}}"
echo "[self-forced-a100x4x2] lr=${{CATK_LR:-preset}} lr_cosine_final_ratio=${{CATK_LR_COSINE_FINAL_RATIO:-preset}} estimator_warmup=${{ESTIMATOR_WARMUP_EPOCHS}} self_forced_use_stop_motion=${{SELF_FORCED_USE_STOP_MOTION}}"
echo "[self-forced-a100x4x2] pretrain_artifact=${{WANDB_PRETRAIN_ARTIFACT}}"
echo "[self-forced-a100x4x2] pretrain_ckpt=${{PRETRAIN_CKPT}}"
echo "[self-forced-a100x4x2] attach survives after exit; press Ctrl-b d to detach"
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


def read_plan_ready(attempt):
    files = sorted(root.glob(attempt + ".*.plan_ready"))
    status_values = []
    for path in files:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
        values = dict(line.split("=", 1) for line in data if "=" in line)
        status_values.append(values.get("status", "1"))
    failure = any(value != "0" for value in status_values)
    return len(files), failure


def read_plan(attempt):
    path = root / (attempt + ".plan")
    if not path.is_file():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    return text, values


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

    def send_file(self, code, path):
        size = path.stat().st_size
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.strip("/")
        if path == "health":
            self.send_text(200, "ok\\n")
            return
        parts = path.split("/", 1)
        if len(parts) != 2 or parts[0] not in (
            "count",
            "aggregate",
            "plan",
            "checkpoint",
            "plan-ready-count",
            "plan-ready-aggregate",
        ):
            self.send_text(404, "not found\\n")
            return
        attempt = parts[1]
        if not name_re.match(attempt):
            self.send_text(400, "bad attempt\\n")
            return
        if parts[0] == "plan":
            text, _ = read_plan(attempt)
            if text is None:
                self.send_text(404, "missing plan\\n")
            else:
                self.send_text(200, text)
            return
        if parts[0] == "checkpoint":
            _, values = read_plan(attempt)
            if not values:
                self.send_text(404, "missing plan\\n")
                return
            if values.get("action") != "fit":
                self.send_text(409, "plan does not require checkpoint sync\\n")
                return
            ckpt_path = pathlib.Path(values.get("ckpt_path", ""))
            if not ckpt_path.is_file():
                self.send_text(404, "missing checkpoint\\n")
                return
            self.send_file(200, ckpt_path)
            return
        if parts[0] in ("plan-ready-count", "plan-ready-aggregate"):
            count, failure = read_plan_ready(attempt)
            if parts[0] == "plan-ready-count":
                self.send_text(200, str(count) + "\\n")
            else:
                self.send_text(
                    200,
                    "count=" + str(count) + "\\n"
                    + "failure=" + ("1" if failure else "0") + "\\n",
                )
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
        if parsed.path not in ("/status", "/plan", "/plan-ready"):
            self.send_text(404, "not found\\n")
            return
        query = urllib.parse.parse_qs(parsed.query)
        attempt = query.get("attempt", [""])[0]
        host = query.get("host", [""])[0]
        if not name_re.match(attempt):
            self.send_text(400, "bad name\\n")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        if parsed.path == "/plan":
            target = root / (attempt + ".plan")
        else:
            if not name_re.match(host):
                self.send_text(400, "bad name\\n")
                return
            suffix = ".status" if parsed.path == "/status" else ".plan_ready"
            target = root / (attempt + "." + host + suffix)
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
  echo "[self-forced-a100x4x2] retry sync server started on $MASTER_ADDR:$RETRY_SYNC_PORT pid=$RETRY_SYNC_PID"
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
      echo "[self-forced-a100x4x2] retry sync server did not become ready at $RETRY_SYNC_HOST:$RETRY_SYNC_PORT" >&2
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
  echo "[self-forced-a100x4x2] failed to post retry status for $attempt_name after retries" >&2
  return 1
}}

post_attempt_plan() {{
  local plan_file="$1"
  local try
  for try in $(seq 1 12); do
    if python - "$attempt_tag" "$plan_file" <<'PY'
import os
import sys
import urllib.parse
import urllib.request

attempt = sys.argv[1]
plan_file = sys.argv[2]
query = urllib.parse.urlencode((("attempt", attempt),))
url = "http://" + os.environ["RETRY_SYNC_HOST"] + ":" + os.environ["RETRY_SYNC_PORT"] + "/plan?" + query
with open(plan_file, "rb") as handle:
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
  echo "[self-forced-a100x4x2] failed to post retry plan for $attempt_tag after retries" >&2
  return 1
}}

fetch_attempt_plan() {{
  local plan_file="$1"
  local tmp_file="${{plan_file}}.$$"
  local waited=0
  local timeout_sec="${{PLAN_SYNC_TIMEOUT_SEC:-600}}"
  while true; do
    if retry_sync_get "plan/$attempt_tag" > "$tmp_file" 2>/dev/null && grep -q '^action=' "$tmp_file"; then
      mv "$tmp_file" "$plan_file"
      return 0
    fi
    rm -f "$tmp_file"
    if (( waited >= timeout_sec )); then
      echo "[self-forced-a100x4x2] timed out waiting for checkpoint plan for $attempt_tag" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
}}

plan_value() {{
  local plan_file="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key {{ print substr($0, index($0, "=") + 1); exit }}' "$plan_file"
}}

checkpoint_matches_plan() {{
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local actual_size=""
  local actual_sha=""

  [[ -f "$path" ]] || return 1
  if [[ -n "$expected_size" ]]; then
    actual_size="$(stat -c %s "$path" 2>/dev/null || true)"
    [[ "$actual_size" == "$expected_size" ]] || return 1
  fi
  if [[ -n "$expected_sha" ]]; then
    actual_sha="$(sha256sum "$path" 2>/dev/null | awk '{{ print $1 }}')"
    [[ "$actual_sha" == "$expected_sha" ]] || return 1
  fi
  return 0
}}

download_plan_checkpoint() {{
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local tmp_file=""

  if checkpoint_matches_plan "$path" "$expected_size" "$expected_sha"; then
    echo "[self-forced-a100x4x2] checkpoint already synced for $attempt_tag: $path"
    return 0
  fi
  if (( NODE_RANK == 0 )); then
    echo "[self-forced-a100x4x2] master checkpoint is missing or does not match plan: $path" >&2
    return 1
  fi

  mkdir -p "$(dirname "$path")"
  tmp_file="${{path}}.download.${{attempt_tag}}.$$"
  rm -f "$tmp_file"
  echo "[self-forced-a100x4x2] downloading resume checkpoint for $attempt_tag from rank0 to $path"
  if ! python - "$attempt_tag" "$tmp_file" <<'PY'
import os
import shutil
import sys
import urllib.parse
import urllib.request

attempt = sys.argv[1]
target = sys.argv[2]
url = (
    "http://"
    + os.environ["RETRY_SYNC_HOST"]
    + ":"
    + os.environ["RETRY_SYNC_PORT"]
    + "/checkpoint/"
    + urllib.parse.quote(attempt)
)
with urllib.request.urlopen(url, timeout=120) as response:
    with open(target, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
PY
  then
    rm -f "$tmp_file"
    return 1
  fi
  if ! checkpoint_matches_plan "$tmp_file" "$expected_size" "$expected_sha"; then
    echo "[self-forced-a100x4x2] downloaded checkpoint failed verification: $tmp_file" >&2
    rm -f "$tmp_file"
    return 1
  fi
  mv -f "$tmp_file" "$path"
  checkpoint_matches_plan "$path" "$expected_size" "$expected_sha"
}}

post_plan_ready() {{
  local status="$1"
  local message="${{2:-ok}}"
  local ready_file="$RETRY_STATE_DIR/$attempt_tag.$(hostname).plan_ready"
  local tmp_file="${{ready_file}}.$$"
  {{
    echo "host=$(hostname)"
    echo "node_rank=$NODE_RANK"
    echo "attempt=$attempt"
    echo "batch_size=$bs"
    echo "status=$status"
    echo "action=${{action:-}}"
    echo "ckpt_path=${{ckpt_path:-}}"
    echo "message=$message"
    echo "time=$(date '+%F %T')"
  }} > "$tmp_file"
  mv "$tmp_file" "$ready_file"
  python - "$attempt_tag" "$ready_file" <<'PY'
import os
import socket
import sys
import urllib.parse
import urllib.request

attempt = sys.argv[1]
ready_file = sys.argv[2]
host = socket.gethostname()
query = urllib.parse.urlencode((("attempt", attempt), ("host", host)))
url = "http://" + os.environ["RETRY_SYNC_HOST"] + ":" + os.environ["RETRY_SYNC_PORT"] + "/plan-ready?" + query
with open(ready_file, "rb") as handle:
    body = handle.read()
request = urllib.request.Request(url, data=body, method="POST")
with urllib.request.urlopen(request, timeout=10) as response:
    response.read()
PY
}}

plan_ready_has_failure() {{
  retry_sync_get "plan-ready-aggregate/$attempt_tag" 2>/dev/null | grep -q '^failure=1$'
}}

wait_for_plan_ready() {{
  local waited=0
  local timeout_sec="${{PLAN_READY_TIMEOUT_SEC:-600}}"
  local count=0
  while true; do
    if plan_ready_has_failure; then
      echo "[self-forced-a100x4x2] checkpoint plan sync failed for $attempt_tag" >&2
      return 1
    fi
    count="$(retry_sync_get "plan-ready-count/$attempt_tag" 2>/dev/null | tr -d '[:space:]')"
    count="${{count:-0}}"
    if (( count >= NNODES )); then
      return 0
    fi
    if (( waited >= timeout_sec )); then
      echo "[self-forced-a100x4x2] timed out waiting for checkpoint plan readiness: got $count/$NNODES" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
}}

resolve_attempt_plan() {{
  local plan_file="$RETRY_STATE_DIR/$attempt_tag.plan"
  local latest_ckpt=""
  local ckpt_size=""
  local ckpt_sha256=""
  local plan_status=0
  local plan_message="ok"

  action=""
  ckpt_path=""

  if (( NODE_RANK == 0 )); then
    latest_ckpt="$(find_latest_self_forced_ckpt)"
    if [[ -n "$latest_ckpt" ]]; then
      action="fit"
      ckpt_path="$latest_ckpt"
      ckpt_size="$(stat -c %s "$ckpt_path" 2>/dev/null || true)"
      ckpt_sha256="$(sha256sum "$ckpt_path" 2>/dev/null | awk '{{ print $1 }}')"
      if [[ -z "$ckpt_size" || -z "$ckpt_sha256" ]]; then
        plan_status=1
        plan_message="failed_to_hash_resume_checkpoint"
      fi
    else
      action="${{INITIAL_ACTION:-auto}}"
      if [[ -z "$action" || "$action" == "auto" ]]; then
        action="finetune"
      fi
      ckpt_path="$PRETRAIN_CKPT"
    fi
    {{
      echo "action=$action"
      echo "ckpt_path=$ckpt_path"
      echo "ckpt_size=$ckpt_size"
      echo "ckpt_sha256=$ckpt_sha256"
      echo "master_host=$(hostname)"
      echo "time=$(date '+%F %T')"
    }} > "$plan_file"
    if ! post_attempt_plan "$plan_file"; then
      plan_status=1
      plan_message="failed_to_publish_plan"
    fi
  else
    if ! fetch_attempt_plan "$plan_file"; then
      plan_status=1
      plan_message="failed_to_fetch_plan"
    fi
  fi

  if (( plan_status == 0 )); then
    action="$(plan_value "$plan_file" action)"
    ckpt_path="$(plan_value "$plan_file" ckpt_path)"
    ckpt_size="$(plan_value "$plan_file" ckpt_size)"
    ckpt_sha256="$(plan_value "$plan_file" ckpt_sha256)"

    if [[ "$action" == "fit" ]]; then
      if ! download_plan_checkpoint "$ckpt_path" "$ckpt_size" "$ckpt_sha256"; then
        plan_status=1
        plan_message="failed_to_sync_resume_checkpoint"
      fi
    elif [[ "$action" == "finetune" ]]; then
      if [[ ! -f "$ckpt_path" ]]; then
        plan_status=1
        plan_message="missing_pretrain_checkpoint"
      fi
    else
      plan_status=1
      plan_message="invalid_plan_action"
    fi
  fi

  if ! post_plan_ready "$plan_status" "$plan_message"; then
    echo "[self-forced-a100x4x2] failed to post checkpoint plan readiness for $attempt_tag" >&2
    return 1
  fi
  if ! wait_for_plan_ready; then
    return 1
  fi
  if (( plan_status != 0 )); then
    echo "[self-forced-a100x4x2] local checkpoint plan sync failed for $attempt_tag: $plan_message" >&2
    return 1
  fi
  echo "[self-forced-a100x4x2] checkpoint plan ready for $attempt_tag: action=$action ckpt=$ckpt_path"
  return 0
}}

ensure_pretrain_checkpoint() {{
  if [[ -f "$PRETRAIN_CKPT" ]]; then
    echo "[self-forced-a100x4x2] using cached pretrain checkpoint: $PRETRAIN_CKPT"
    return 0
  fi

  mkdir -p "$(dirname "$PRETRAIN_CKPT")" "$WANDB_PRETRAIN_DOWNLOAD_DIR"
  lock_dir="${{PRETRAIN_CKPT}}.download.lock"

  if mkdir "$lock_dir" 2>/dev/null; then
    echo "[self-forced-a100x4x2] downloading W&B artifact: $WANDB_PRETRAIN_ARTIFACT"
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
      echo "[self-forced-a100x4x2] W&B artifact download failed with status $status" >&2
      return "$status"
    fi
  else
    echo "[self-forced-a100x4x2] waiting for checkpoint download lock: $lock_dir"
    for _ in $(seq 1 180); do
      if [[ -f "$PRETRAIN_CKPT" ]]; then
        echo "[self-forced-a100x4x2] checkpoint appeared: $PRETRAIN_CKPT"
        return 0
      fi
      sleep 10
    done
    echo "[self-forced-a100x4x2] timed out waiting for $PRETRAIN_CKPT" >&2
    return 4
  fi

  test -f "$PRETRAIN_CKPT"
}}

find_latest_self_forced_ckpt() {{
  {{
    ls -t "$LOG_DIR/$TASK_NAME/runs"/*/checkpoints/epoch_last.ckpt 2>/dev/null
    ls -t "$LOG_DIR/$TASK_NAME/runs"/*/checkpoints/last.ckpt 2>/dev/null
  }} | head -1
}}

strip_shell_quotes() {{
  local value="$1"
  value="${{value%\\\"}}"
  value="${{value#\\\"}}"
  value="${{value%\'}}"
  value="${{value#\'}}"
  printf '%s\n' "$value"
}}

resolve_generated_estimator_lr_config() {{
  if [[ -n "${{CATK_GENERATED_ESTIMATOR_LR:-}}" ]]; then
    strip_shell_quotes "$CATK_GENERATED_ESTIMATOR_LR"
    return 0
  fi

  local value=""
  local token
  if [[ -n "${{CATK_EXTRA_OVERRIDES:-}}" ]]; then
    for token in $CATK_EXTRA_OVERRIDES; do
      case "$token" in
        model.model_config.self_forced.generated_estimator_lr=*|+model.model_config.self_forced.generated_estimator_lr=*)
          value="${{token#*=}}"
          ;;
      esac
    done
  fi

  if [[ -n "$value" ]]; then
    strip_shell_quotes "$value"
    return 0
  fi

  strip_shell_quotes "${{CATK_LR:-}}"
}}

apply_estimator_warmup_bank_progress() {{
  local loaded_warmup="$1"
  local remaining_warmup="$2"
  local context="$3"

  ESTIMATOR_WARMUP_BANK_LOADED_WARMUP="$loaded_warmup"
  ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="$remaining_warmup"
  if [[ "$ESTIMATOR_WARMUP_BANK_ADJUST_MAX_EPOCHS" == "true" && "${{MAX_EPOCHS:-}}" =~ ^[0-9]+$ ]]; then
    local adjusted_epochs=$(( MAX_EPOCHS - ESTIMATOR_WARMUP_BANK_LOADED_WARMUP ))
    if (( adjusted_epochs < 1 )); then
      adjusted_epochs=1
    fi
    echo "[self-forced-a100x4x2] adjusting MAX_EPOCHS ${{MAX_EPOCHS}} -> ${{adjusted_epochs}} after estimator-bank $context"
    MAX_EPOCHS="$adjusted_epochs"
  fi
  ESTIMATOR_WARMUP_EPOCHS="$ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP"
}}

maybe_prepare_estimator_warmup_bank() {{
  if [[ "${{ESTIMATOR_WARMUP_BANK_ENABLED:-false}}" != "true" ]]; then
    return 0
  fi
  if [[ -z "${{ESTIMATOR_WARMUP_BANK_ARTIFACT:-}}" ]]; then
    echo "[self-forced-a100x4x2] estimator warmup bank enabled but artifact is empty; running warmup normally"
    return 0
  fi
  if [[ -z "${{ESTIMATOR_WARMUP_EPOCHS:-}}" || "${{ESTIMATOR_WARMUP_EPOCHS}}" == "0" ]]; then
    return 0
  fi
  ESTIMATOR_WARMUP_BANK_LR="$(resolve_generated_estimator_lr_config)"
  if [[ -z "${{ESTIMATOR_WARMUP_BANK_LR:-}}" ]]; then
    echo "[self-forced-a100x4x2] estimator warmup bank enabled but generated estimator lr is empty; running warmup normally"
    return 0
  fi
  if [[ -n "$(find_latest_self_forced_ckpt)" ]]; then
    echo "[self-forced-a100x4x2] existing self-forced checkpoint found; keeping resume path and skipping estimator-bank resolve"
    return 0
  fi

  local bank_root="$RUN_ROOT/estimator_bank/$(hostname)"
  mkdir -p "$bank_root"
  local requested_warmup="${{ESTIMATOR_WARMUP_EPOCHS}}"
  local resolved_env="$bank_root/resolved_warmup_${{requested_warmup}}_lr_${{ESTIMATOR_WARMUP_BANK_LR}}.env"
  ESTIMATOR_WARMUP_BANK_INIT_PATH="$bank_root/resolved_for_warmup_${{requested_warmup}}_lr_${{ESTIMATOR_WARMUP_BANK_LR}}_generated_estimator.pt"
  ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH="$bank_root/snapshot_warmup_${{requested_warmup}}_lr_${{ESTIMATOR_WARMUP_BANK_LR}}_generated_estimator.pt"

  echo "[self-forced-a100x4x2] checking estimator warmup bank: artifact=${{ESTIMATOR_WARMUP_BANK_ARTIFACT}} requested_warmup=${{requested_warmup}} generated_estimator_lr=${{ESTIMATOR_WARMUP_BANK_LR}}"
  if python scripts/self_forced_estimator_bank.py resolve \
      --artifact "$ESTIMATOR_WARMUP_BANK_ARTIFACT" \
      --warmup-epochs "$requested_warmup" \
      --lr "$ESTIMATOR_WARMUP_BANK_LR" \
      --output "$ESTIMATOR_WARMUP_BANK_INIT_PATH" \
      --env-output "$resolved_env" \
      --entity "$ESTIMATOR_WARMUP_BANK_ENTITY" \
      --project "$ESTIMATOR_WARMUP_BANK_PROJECT"; then
    # shellcheck disable=SC1090
    source "$resolved_env"
    local loaded_warmup="${{ESTIMATOR_WARMUP_BANK_RESOLVED_WARMUP:-0}}"
    local remaining_warmup="${{ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP:-0}}"
    echo "[self-forced-a100x4x2] estimator bank hit: loaded_warmup=${{loaded_warmup}} requested_warmup=${{requested_warmup}} remaining_warmup=${{remaining_warmup}}"
    apply_estimator_warmup_bank_progress "$loaded_warmup" "$remaining_warmup" "hit"
    if [[ "$remaining_warmup" == "0" ]]; then
      ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH=""
    fi
  else
    echo "[self-forced-a100x4x2] estimator bank miss; warmup will run and snapshot will be saved to $ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH"
    ESTIMATOR_WARMUP_BANK_LOADED_WARMUP=0
    ESTIMATOR_WARMUP_BANK_REMAINING_WARMUP="$requested_warmup"
  fi
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
    if global_attempt_has_oom; then
      echo "[self-forced-a100x4x2] OOM status observed for attempt $attempt; skipping full status barrier"
      return 0
    fi
    count="$(retry_sync_get "count/$attempt_tag" 2>/dev/null | tr -d '[:space:]')"
    count="${{count:-0}}"
    if (( count >= NNODES )); then
      return 0
    fi
    if (( waited >= timeout_sec )); then
      echo "[self-forced-a100x4x2] timed out waiting for attempt $attempt status files: got $count/$NNODES" >&2
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

terminate_process_group() {{
  local pgid="$1"
  if [[ -z "$pgid" || "$pgid" == "0" ]]; then
    return 0
  fi
  kill -TERM -- "-$pgid" 2>/dev/null || true
  sleep "${{OOM_WATCH_KILL_GRACE_SEC:-20}}"
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

  echo "[self-forced-a100x4x2] terminating task processes for $reason: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  while (( waited < grace_sec )); do
    sleep 1
    waited=$(( waited + 1 ))
    mapfile -t pids < <(task_process_pids || true)
    if (( ${{#pids[@]}} == 0 )); then
      return 0
    fi
  done

  echo "[self-forced-a100x4x2] force killing task processes for $reason: ${{pids[*]}}"
  kill -KILL "${{pids[@]}}" 2>/dev/null || true
}}

terminate_attempt_processes() {{
  local pgid_file="$1"
  local reason="$2"
  local pgid=""

  pgid="$(cat "$pgid_file" 2>/dev/null || true)"
  terminate_process_group "$pgid"
  terminate_task_processes "$reason"
}}

run_torchrun_attempt() {{
  local torch_status_file="$RUN_ROOT/$(hostname).${{attempt_tag}}.torchrun_status"
  local torch_pgid_file="$RUN_ROOT/$(hostname).${{attempt_tag}}.torchrun_pgid"
  local oom_watch_file="$RUN_ROOT/$(hostname).${{attempt_tag}}.remote_oom_watch"
  local tee_pid=""
  local watch_pid=""
  local pgid=""
  local status="1"

  rm -f "$torch_status_file" "$torch_pgid_file" "$oom_watch_file"
  terminate_task_processes "pre-attempt stale cleanup for $attempt_tag"

  (
    set +e
    setsid bash -c 'pgid_file="$1"; shift; echo "$$" > "$pgid_file"; exec "$@"' \
      bash "$torch_pgid_file" torchrun "${{torchrun_args[@]}}"
    echo "$?" > "$torch_status_file"
  ) 2>&1 | tee "$attempt_log" &
  tee_pid=$!

  (
    set +e
    while kill -0 "$tee_pid" 2>/dev/null; do
      if global_attempt_has_oom; then
        echo "[self-forced-a100x4x2] remote OOM observed for $attempt_tag; terminating local torchrun group"
        terminate_attempt_processes "$torch_pgid_file" "remote OOM on $attempt_tag"
        touch "$oom_watch_file"
        exit 0
      fi
      sleep "${{OOM_WATCH_INTERVAL_SEC:-5}}"
    done
  ) &
  watch_pid=$!

  wait "$tee_pid"
  if [[ -n "$watch_pid" ]]; then
    kill "$watch_pid" 2>/dev/null || true
    wait "$watch_pid" 2>/dev/null || true
  fi

  if [[ -f "$torch_status_file" ]]; then
    status="$(cat "$torch_status_file" 2>/dev/null || echo 1)"
  fi
  if [[ -f "$oom_watch_file" && "$status" == "0" ]]; then
    status="1"
  fi
  if [[ "$status" != "0" || -f "$oom_watch_file" ]]; then
    terminate_attempt_processes "$torch_pgid_file" "post-attempt cleanup for $attempt_tag status=$status"
  fi
  return "$status"
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

  if ! resolve_attempt_plan; then
    echo "[self-forced-a100x4x2] refusing to start $attempt_tag because checkpoint/action plan is not synchronized" >&2
    exit 1
  fi
  if (( attempt == 1 )) && [[ "$action" == "finetune" ]]; then
    maybe_prepare_estimator_warmup_bank || exit $?
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
    model.model_config.self_forced.use_stop_motion="$SELF_FORCED_USE_STOP_MOTION"
  )

  if [[ -n "${{CATK_LR:-}}" ]]; then
    torchrun_args+=(model.model_config.lr="$CATK_LR")
  fi
  if [[ -n "${{CATK_LR_COSINE_FINAL_RATIO:-}}" ]]; then
    torchrun_args+=(model.model_config.self_forced.lr_cosine_final_ratio="$CATK_LR_COSINE_FINAL_RATIO")
  fi
  if [[ -n "${{DECODER_USE_STOP_MOTION:-}}" ]]; then
    torchrun_args+=(model.model_config.decoder.use_stop_motion="$DECODER_USE_STOP_MOTION")
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
  if [[ -n "${{CHECK_VAL_EVERY_N_EPOCH:-}}" ]]; then
    torchrun_args+=(trainer.check_val_every_n_epoch="$CHECK_VAL_EVERY_N_EPOCH")
  fi
  if [[ -n "${{TRAIN_EPOCH_SAMPLE_FRACTION:-}}" ]]; then
    torchrun_args+=(data.train_epoch_sample_fraction="$TRAIN_EPOCH_SAMPLE_FRACTION")
  fi
  if [[ -n "${{TRAIN_MEMORY_BALANCED_BATCHES:-}}" ]]; then
    torchrun_args+=(data.train_memory_balanced_batches="$TRAIN_MEMORY_BALANCED_BATCHES")
  fi
  if [[ "$action" == "finetune" && -n "${{ESTIMATOR_WARMUP_BANK_INIT_PATH:-}}" && -f "$ESTIMATOR_WARMUP_BANK_INIT_PATH" ]]; then
    torchrun_args+=(model.model_config.self_forced.generated_estimator_init_path="$ESTIMATOR_WARMUP_BANK_INIT_PATH")
    if [[ "${{ESTIMATOR_WARMUP_EPOCHS:-0}}" == "0" ]]; then
      torchrun_args+=(model.model_config.self_forced.generated_estimator_skip_warmup_on_load=true)
    else
      torchrun_args+=(model.model_config.self_forced.generated_estimator_skip_warmup_on_load=false)
    fi
  fi
  if [[ -n "${{ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH:-}}" && "${{ESTIMATOR_WARMUP_BANK_ENABLED:-false}}" == "true" ]]; then
    torchrun_args+=(model.model_config.self_forced.generated_estimator_bank_snapshot_path="$ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH")
  fi
  if [[ -n "${{ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP:-}}" && "${{ESTIMATOR_WARMUP_BANK_ENABLED:-false}}" == "true" ]]; then
    torchrun_args+=(model.model_config.self_forced.generated_estimator_bank_target_warmup_epochs="$ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP")
    torchrun_args+=(model.model_config.self_forced.generated_estimator_bank_loaded_warmup_epochs="$ESTIMATOR_WARMUP_BANK_LOADED_WARMUP")
  fi
  torchrun_args+=("${{extra_overrides[@]}}")
  if [[ -n "${{CATK_GENERATED_ESTIMATOR_LR:-}}" ]]; then
    torchrun_args+=(model.model_config.self_forced.generated_estimator_lr="$(strip_shell_quotes "$CATK_GENERATED_ESTIMATOR_LR")")
  fi

  echo
  echo "[self-forced-a100x4x2] $attempt_tag action=$action ckpt=$ckpt_path"
  printf '[self-forced-a100x4x2] torchrun'
  printf ' %q' "${{torchrun_args[@]}}"
  printf '\\n'

  run_torchrun_attempt
  status=$?
  local_oom=0
  if grep -Eq "$OOM_REGEX" "$attempt_log" 2>/dev/null; then
    local_oom=1
  fi
  if [[ -f "$RUN_ROOT/$(hostname).${{attempt_tag}}.remote_oom_watch" ]]; then
    local_oom=1
  fi
  write_attempt_status "$status" "$local_oom"

  if ! wait_for_attempt_statuses; then
    echo "[self-forced-a100x4x2] retry barrier failed for $attempt_tag; see $RETRY_STATE_DIR" >&2
    exit 1
  fi

  if ! global_attempt_has_failure; then
    echo "[self-forced-a100x4x2] training completed successfully at bs=$bs"
    if (( NODE_RANK == 0 )) \
        && [[ "${{ESTIMATOR_WARMUP_BANK_ENABLED:-false}}" == "true" ]] \
        && [[ -n "${{ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME:-}}" ]] \
        && [[ -n "${{ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH:-}}" ]] \
        && [[ -f "$ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH" ]] \
        && [[ -n "${{ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP:-}}" ]] \
        && [[ -n "${{ESTIMATOR_WARMUP_BANK_LR:-}}" ]]; then
      echo "[self-forced-a100x4x2] uploading generated-estimator warmup snapshot to W&B bank: $ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME generated_estimator_lr=$ESTIMATOR_WARMUP_BANK_LR"
      python scripts/self_forced_estimator_bank.py upsert \
        --artifact-name "$ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME" \
        --entry "${{ESTIMATOR_WARMUP_BANK_ORIGINAL_WARMUP}}:${{ESTIMATOR_WARMUP_BANK_LR}}:${{ESTIMATOR_WARMUP_BANK_SNAPSHOT_PATH}}" \
        --entity "$ESTIMATOR_WARMUP_BANK_ENTITY" \
        --project "$ESTIMATOR_WARMUP_BANK_PROJECT" \
        --run-name "${{TASK_NAME}}_generated_estimator_bank"
    fi
    exit 0
  fi

  if global_attempt_has_oom; then
    next_bs=$(( bs - OOM_STEP ))
    echo "[self-forced-a100x4x2] OOM detected on at least one node in attempt $attempt; all nodes lowering bs $bs -> $next_bs"
    terminate_task_processes "OOM retry cleanup before next attempt"
    sleep "${{OOM_RETRY_RESTART_GRACE_SEC:-10}}"
    bs="$next_bs"
    continue
  fi

  echo "[self-forced-a100x4x2] non-OOM failure status=$status; see $attempt_log and $RETRY_STATE_DIR/$attempt_tag.*.status"
  exit "$status"
done

echo "[self-forced-a100x4x2] reached MIN_BS=$MIN_BS without success"
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
TASK_NAME_TO_STOP={shq(args.task_name)}
mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
if (( ${{#pids[@]}} > 0 )); then
  echo "[launcher] terminating stale task processes for $TASK_NAME_TO_STOP before replace: ${{pids[*]}}"
  kill -TERM "${{pids[@]}}" 2>/dev/null || true
  sleep 10
  mapfile -t pids < <(pgrep -f "task_name=${{TASK_NAME_TO_STOP}}" 2>/dev/null || true)
  if (( ${{#pids[@]}} > 0 )); then
    echo "[launcher] force killing stale task processes for $TASK_NAME_TO_STOP before replace: ${{pids[*]}}"
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
        description="Launch self-forced A100x4x2 training on existing static pods.",
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
    parser.add_argument("--master-port", default="29583")
    parser.add_argument(
        "--retry-sync-port",
        default="29584",
        help=(
            "Rank-0 pod HTTP port used to collect retry status, publish the "
            "shared checkpoint/action plan, and serve resume checkpoints."
        ),
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--initial-bs", type=int, default=96)
    parser.add_argument("--oom-step", type=int, default=2)
    parser.add_argument("--min-bs", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--scorer-scene-num", type=int, default=1680)
    parser.add_argument("--unfrozen-range", default="middle")
    parser.add_argument("--estimator-warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--use-estimator-warmup-bank",
        dest="use_estimator_warmup_bank",
        action="store_true",
        help="Enable the shared generated-estimator warmup W&B bank. This is the default.",
    )
    parser.add_argument(
        "--no-estimator-warmup-bank",
        dest="use_estimator_warmup_bank",
        action="store_false",
        help="Disable the shared generated-estimator warmup W&B bank for this run.",
    )
    parser.add_argument(
        "--estimator-warmup-bank-artifact",
        default=DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT,
    )
    parser.add_argument(
        "--estimator-warmup-bank-artifact-name",
        default=DEFAULT_ESTIMATOR_WARMUP_BANK_ARTIFACT_NAME,
    )
    parser.add_argument("--estimator-warmup-bank-entity", default="jksg01019-naver-labs")
    parser.add_argument("--estimator-warmup-bank-project", default="SMART-FLOW")
    parser.add_argument(
        "--no-estimator-warmup-bank-adjust-max-epochs",
        dest="estimator_warmup_bank_adjust_max_epochs",
        action="store_false",
    )
    parser.set_defaults(estimator_warmup_bank_adjust_max_epochs=True)
    parser.set_defaults(use_estimator_warmup_bank=True)
    parser.add_argument("--self-forced-use-stop-motion", default="false")
    parser.add_argument("--decoder-use-stop-motion", default="")
    parser.add_argument(
        "--initial-action",
        choices=("auto", "finetune", "fit"),
        default="auto",
        help=(
            "Action to use when no task checkpoint exists yet. Default auto keeps "
            "the historical finetune behavior; use fit for full Lightning checkpoint resumes."
        ),
    )
    parser.add_argument("--learning-rate", default="1.0e-6")
    parser.add_argument(
        "--generated-estimator-learning-rate",
        default="",
        help="Generated-estimator optimizer lr. Defaults to --learning-rate via model config.",
    )
    parser.add_argument(
        "--lr-cosine-final-ratio",
        default="",
        help="Optional final cosine LR multiplier override for self-forced optimizers.",
    )
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--max-epochs", default="")
    parser.add_argument("--check-val-every-n-epoch", default="")
    parser.add_argument("--train-epoch-sample-fraction", default="")
    parser.add_argument("--train-memory-balanced-batches", default="true")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--monitor-interval", type=int, default=30)
    parser.add_argument("--no-monitor-pane", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stop:
        return args
    if len(args.pods) < 1:
        parser.error("--pods must contain at least one pod")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be >= 1")
    if args.initial_bs < 1:
        parser.error("--initial-bs must be >= 1")
    if args.oom_step < 1:
        parser.error("--oom-step must be >= 1")
    if args.min_bs < 1:
        parser.error("--min-bs must be >= 1")
    if args.self_forced_use_stop_motion not in {"true", "false"}:
        parser.error("--self-forced-use-stop-motion must be 'true' or 'false'")
    if args.decoder_use_stop_motion not in {"", "true", "false"}:
        parser.error("--decoder-use-stop-motion must be empty, 'true', or 'false'")
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
            exec_in_pod(args, pod, render_stop_command(args.session, args.task_name))
        return

    master_addr = args.master_addr or (
        "<MASTER_POD_IP>" if args.dry_run else pod_ip(args.namespace, args.pods[0])
    )
    print(f"[launcher] master pod: {args.pods[0]} ({master_addr}:{args.master_port})")
    print(f"[launcher] retry sync: {master_addr}:{args.retry_sync_port}")
    print(f"[launcher] task_name: {args.task_name}")
    print(f"[launcher] session:   {args.session}")
    print(f"[launcher] artifact:  {args.wandb_pretrain_artifact}")
    print(f"[launcher] ckpt path: {args.pretrain_ckpt}")
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
