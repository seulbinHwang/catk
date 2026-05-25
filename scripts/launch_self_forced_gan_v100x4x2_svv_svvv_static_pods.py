#!/usr/bin/env python3
"""Launch Set-level Self-Forced GAN fine-tuning on static pods.

This launcher defaults to two already-running static V100x4 pods. Pod-specific
entrypoints may override the defaults before calling ``main()``. It never
creates, deletes, or restarts pods. It prepares the W&B pretrain checkpoint,
optionally builds/syncs the offline teacher rollout cache, then delegates the
actual tmux run to ``launch_h100x4_multinode_pretrain_tmux.py``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


BASE_LAUNCHER = Path(__file__).with_name("launch_h100x4_multinode_pretrain_tmux.py")

DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_PODS = ("svv", "svvv")
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_BRANCH = "semi_control_rolling_gan"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_TEACHER_CACHE_ROOT = "/workspace/womd_v1_3/SMART_teacher_gan_cache_h100x6_bs18_latest"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_EXPERIMENT = "self_forced_gan_v100x4x2_svv_svvv"
DEFAULT_TASK_NAME = "sf_gan_k16_v100x4x2_svv_svvv"
DEFAULT_SESSION = "catk-sf-gan-v100x4x2-svv-svvv"
DEFAULT_DESCRIPTION = "Launch Set-level Self-Forced GAN fine-tuning on svv + svvv V100x4 pods."
DEFAULT_EXPECTED_POD_COUNT = 2
DEFAULT_MASTER_PORT = "29670"
DEFAULT_CHECKPOINT_SYNC_PORT = "29671"
DEFAULT_NPROC_PER_NODE = 4
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_ACCUMULATE_GRAD_BATCHES = 12
DEFAULT_VAL_BATCH_SIZE = 2
DEFAULT_PRECISION = "16-mixed"
DEFAULT_TEACHER_CACHE_GPUS_PER_POD = 4
DEFAULT_TEACHER_CACHE_BATCH_SIZE = 32
DEFAULT_TEACHER_CACHE_ROLLOUT_BATCH_SIZE = 32
DEFAULT_TEACHER_CACHE_DATA_NUM_WORKERS = 0
DEFAULT_TEACHER_CACHE_DATA_PREFETCH_FACTOR = 2
DEFAULT_TEACHER_CACHE_SAVE_WORKERS = 4
DEFAULT_TEACHER_CACHE_AMP_DTYPE = "float16"
DEFAULT_WANDB_ENTITY = "jksg01019-naver-labs"
DEFAULT_WANDB_PROJECT = "SMART-FLOW"
DEFAULT_WANDB_PRETRAIN_TASK = (
    "flow_control_space_pretrain_h100x6_hsb2_wo1_execctx_prefix_balanced_"
    "lr6e-4_bs18_oomretry"
)
DEFAULT_WANDB_PRETRAIN_ARTIFACT = "epoch-last-sqverrgj:v38"
DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_control_space_pretrain_h100x6_hsb2_wo1_execctx_prefix_"
    "balanced_lr6e-4_bs18_oomretry/latest/epoch_last.ckpt"
)
DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_control_space_pretrain_h100x6_hsb2_wo1_execctx_prefix_"
    "balanced_lr6e-4_bs18_oomretry/latest/artifact"
)


def shq(value: object) -> str:
    return shlex.quote(str(value))


def run_kubectl(args: list[str], *, stdin=None, stdout=None, capture: bool = False) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        text=False if stdin is not None or stdout is not None else True,
        stdin=stdin,
        stdout=subprocess.PIPE if capture else stdout,
    )
    if capture:
        output = result.stdout
        if isinstance(output, bytes):
            return output.decode("utf-8").strip()
        return str(output).strip()
    return ""


def get_pod_ip(args: argparse.Namespace, pod: str) -> str:
    if args.dry_run:
        return "127.0.0.1"
    return run_kubectl(
        [
            "get",
            "pod",
            "-n",
            args.namespace,
            pod,
            "-o",
            "jsonpath={.status.podIP}",
        ],
        capture=True,
    )


def exec_in_pod(args: argparse.Namespace, pod: str, script: str, *, capture: bool = False) -> str:
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
        return ""
    return run_kubectl(command, capture=capture)


def popen_exec_in_pod(args: argparse.Namespace, pod: str, script: str) -> subprocess.Popen:
    command = [
        "kubectl",
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
        print(" ".join(shq(part) for part in command))
        return subprocess.Popen(["true"])
    return subprocess.Popen(command)


def normalize_artifact_ref(args: argparse.Namespace, value: str) -> str:
    if not value:
        return ""
    if "/" in value:
        return value
    return f"{args.wandb_entity}/{args.wandb_project}/{value}"


def render_pretrain_download_script(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
export WANDB_ENTITY={shq(args.wandb_entity)}
export WANDB_PROJECT={shq(args.wandb_project)}
export WANDB_PRETRAIN_TASK={shq(args.wandb_pretrain_task)}
export WANDB_PRETRAIN_ARTIFACT={shq(normalize_artifact_ref(args, args.wandb_pretrain_artifact))}
export PRETRAIN_CKPT={shq(args.pretrain_ckpt)}
export PRETRAIN_DOWNLOAD_DIR={shq(args.pretrain_download_dir)}
export FORCE_PRETRAIN_DOWNLOAD={shq("1" if args.force_pretrain_download else "0")}
python - <<'PY'
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import wandb

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
task_name = os.environ["WANDB_PRETRAIN_TASK"]
artifact_ref = os.environ["WANDB_PRETRAIN_ARTIFACT"].strip()
target_ckpt = Path(os.environ["PRETRAIN_CKPT"])
download_dir = Path(os.environ["PRETRAIN_DOWNLOAD_DIR"])
force = os.environ.get("FORCE_PRETRAIN_DOWNLOAD") == "1"
marker_path = target_ckpt.with_suffix(target_ckpt.suffix + ".wandb.json")

def expected_marker_from_artifact_ref(value):
    if not value:
        return None
    artifact_name = value.split("/")[-1]
    if ":" not in artifact_name:
        return None
    version = artifact_name.rsplit(":", 1)[1]
    return {{
        "artifact": artifact_name,
        "version": version,
        "qualified_artifact": value,
    }}

expected_marker = expected_marker_from_artifact_ref(artifact_ref)
if expected_marker and target_ckpt.is_file() and marker_path.is_file() and not force:
    try:
        current = json.loads(marker_path.read_text())
    except Exception:
        current = {{}}
    if (
        current.get("artifact") == expected_marker["artifact"]
        and current.get("version") == expected_marker["version"]
    ):
        checkpoint = torch.load(target_ckpt, map_location="cpu", weights_only=False)
        print(
            "[pretrain] using cached pinned W&B checkpoint "
            f"artifact={{expected_marker['artifact']}} epoch={{checkpoint.get('epoch')}} "
            f"global_step={{checkpoint.get('global_step')}} path={{target_ckpt}}",
            flush=True,
        )
        raise SystemExit(0)

api = wandb.Api(timeout=60)
if artifact_ref:
    artifact = api.artifact(artifact_ref)
    run_id = None
else:
    filters = {{"$or": [{{"display_name": task_name}}, {{"config.task_name": task_name}}]}}
    runs = api.runs(f"{{entity}}/{{project}}", filters=filters, order="-created_at", per_page=10)
    if not runs:
        raise SystemExit(f"no W&B run found for task_name/display_name={{task_name!r}}")
    run = runs[0]
    run_id = run.id
    candidates = []
    for item in run.logged_artifacts():
        if item.type != "model":
            continue
        if not item.name.startswith(f"epoch-last-{{run.id}}:"):
            continue
        match = re.search(r":v(\\d+)$", item.name)
        version = int(match.group(1)) if match else -1
        candidates.append((version, item))
    if not candidates:
        raise SystemExit(f"run {{run.id}} has no epoch-last model artifacts")
    artifact = sorted(candidates, key=lambda pair: pair[0])[-1][1]

resolved = {{
    "artifact": artifact.name,
    "version": artifact.version,
    "qualified_artifact": f"{{entity}}/{{project}}/{{artifact.name}}",
    "source_task": task_name,
    "source_run_id": run_id,
}}

if target_ckpt.is_file() and marker_path.is_file() and not force:
    try:
        current = json.loads(marker_path.read_text())
    except Exception:
        current = {{}}
    if (
        current.get("artifact") == resolved["artifact"]
        and current.get("version") == resolved["version"]
    ):
        checkpoint = torch.load(target_ckpt, map_location="cpu", weights_only=False)
        mode = "pinned" if artifact_ref else "latest"
        print(
            f"[pretrain] using cached {{mode}} W&B checkpoint "
            f"artifact={{artifact.name}} epoch={{checkpoint.get('epoch')}} "
            f"global_step={{checkpoint.get('global_step')}} path={{target_ckpt}}",
            flush=True,
        )
        raise SystemExit(0)

download_dir.mkdir(parents=True, exist_ok=True)
target_ckpt.parent.mkdir(parents=True, exist_ok=True)
artifact_dir = Path(artifact.download(root=str(download_dir)))
candidates = []
preferred = artifact_dir / "epoch_last.ckpt"
if preferred.is_file():
    candidates.append(preferred.as_posix())
candidates.extend(glob.glob(str(artifact_dir / "**" / "epoch_last.ckpt"), recursive=True))
candidates.extend(glob.glob(str(artifact_dir / "**" / "*.ckpt"), recursive=True))
candidates = list(dict.fromkeys(candidates))
if not candidates:
    raise SystemExit(f"no checkpoint file found in downloaded artifact dir: {{artifact_dir}}")

tmp_path = target_ckpt.with_suffix(target_ckpt.suffix + ".tmp")
shutil.copy2(candidates[0], tmp_path)
os.replace(tmp_path, target_ckpt)
checkpoint = torch.load(target_ckpt, map_location="cpu", weights_only=False)
resolved.update({{"checkpoint_epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}})
marker_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\\n")
print(
    "[pretrain] downloaded W&B checkpoint "
    f"artifact={{artifact.name}} epoch={{checkpoint.get('epoch')}} "
    f"global_step={{checkpoint.get('global_step')}} path={{target_ckpt}}",
    flush=True,
)
PY
"""


def render_teacher_cache_key_script(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
python - <<'PY'
import json
import re
from pathlib import Path

import torch

ckpt_path = Path({args.pretrain_ckpt!r})
marker_path = ckpt_path.with_suffix(ckpt_path.suffix + ".wandb.json")
marker = {{}}
if marker_path.exists():
    try:
        marker = json.loads(marker_path.read_text())
    except Exception:
        marker = {{}}
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
artifact = marker.get("artifact") or marker.get("qualified_artifact") or ckpt_path.stem
artifact = artifact.split("/")[-1]
artifact_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(artifact)).strip("._-") or "checkpoint"
epoch = checkpoint.get("epoch")
global_step = checkpoint.get("global_step")
parts = [
    artifact_key,
    f"epoch{{epoch}}" if epoch is not None else "epoch_unknown",
    f"gs{{global_step}}" if global_step is not None else "gs_unknown",
    f"seed{int(args.teacher_cache_seed)}",
    f"k{int(args.teacher_cache_rollouts)}",
    "fp16",
]
if {bool(args.teacher_cache_max_scenes)!r}:
    parts.append({("max" + str(args.teacher_cache_max_scenes))!r})
print("_".join(parts))
PY
"""


def resolve_teacher_cache_root(args: argparse.Namespace) -> None:
    if not args.teacher_cache_keyed_root:
        return
    if args.dry_run:
        key = "checkpoint_keyed_teacher_cache"
    else:
        key = exec_in_pod(args, args.pods[0], render_teacher_cache_key_script(args), capture=True).strip()
    if not key:
        raise RuntimeError("failed to resolve teacher cache key")
    base = Path(args.teacher_cache_root)
    if base.name != key:
        args.teacher_cache_root = str(base / key)
    print(f"[launcher] resolved checkpoint-keyed teacher cache root: {args.teacher_cache_root}", flush=True)


def render_teacher_cache_build_script(args: argparse.Namespace, *, max_scenes: str) -> str:
    max_arg = f" --max-scenes {shq(max_scenes)}" if max_scenes else ""
    skip_arg = " --skip-existing" if args.skip_existing_teacher_cache else ""
    batch_arg = f" --batch-size {int(args.teacher_cache_batch_size)}"
    rollout_batch_arg = f" --rollout-batch-size {int(args.teacher_cache_rollout_batch_size)}"
    data_workers_arg = f" --data-num-workers {int(args.teacher_cache_data_num_workers)}"
    prefetch_arg = f" --data-prefetch-factor {int(args.teacher_cache_data_prefetch_factor)}"
    save_workers_arg = f" --save-workers {int(args.teacher_cache_save_workers)}"
    amp_dtype_arg = f" --amp-dtype {shq(args.teacher_cache_amp_dtype)}"
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
export CACHE_ROOT={shq(args.cache_root)}
export TEACHER_GAN_CACHE_ROOT={shq(args.teacher_cache_root)}
mkdir -p "$TEACHER_GAN_CACHE_ROOT"
PYTHONPATH=. python tools/build_self_forced_gan_teacher_cache.py \
  --ckpt-path {shq(args.pretrain_ckpt)} \
  --output-root "$TEACHER_GAN_CACHE_ROOT" \
  --split train \
  --rollouts-per-scene {int(args.teacher_cache_rollouts)} \
  --seed {int(args.teacher_cache_seed)} \
  --storage-dtype float16 \
  {amp_dtype_arg} \
  --device cuda:0{max_arg}{skip_arg} \
  {batch_arg} \
  {rollout_batch_arg} \
  {data_workers_arg} \
  {prefetch_arg} \
  {save_workers_arg} \
  --override paths.cache_root="$CACHE_ROOT" \
  --override experiment={shq(args.experiment)}
PYTHONPATH=. python tools/validate_self_forced_gan_cache.py "$TEACHER_GAN_CACHE_ROOT" --max-files {int(args.validate_cache_max_files)}
"""


def render_parallel_teacher_cache_build_script(
    args: argparse.Namespace,
    *,
    pod_index: int,
    num_shards: int,
    max_scenes: str,
) -> str:
    max_arg = f" --max-scenes {shq(max_scenes)}" if max_scenes else ""
    skip_arg = " --skip-existing" if args.skip_existing_teacher_cache else ""
    commands = []
    for local_gpu in range(int(args.teacher_cache_gpus_per_pod)):
        shard_index = int(pod_index) * int(args.teacher_cache_gpus_per_pod) + local_gpu
        commands.append(
            " ".join(
                [
                    f"CUDA_VISIBLE_DEVICES={local_gpu}",
                    "PYTHONPATH=.",
                    "python tools/build_self_forced_gan_teacher_cache.py",
                    f"--ckpt-path {shq(args.pretrain_ckpt)}",
                    f"--output-root \"$TEACHER_GAN_CACHE_ROOT\"",
                    "--split train",
                    f"--rollouts-per-scene {int(args.teacher_cache_rollouts)}",
                    f"--seed {int(args.teacher_cache_seed)}",
                    "--storage-dtype float16",
                    f"--amp-dtype {shq(args.teacher_cache_amp_dtype)}",
                    "--device cuda:0",
                    f"--batch-size {int(args.teacher_cache_batch_size)}",
                    f"--rollout-batch-size {int(args.teacher_cache_rollout_batch_size)}",
                    f"--data-num-workers {int(args.teacher_cache_data_num_workers)}",
                    f"--data-prefetch-factor {int(args.teacher_cache_data_prefetch_factor)}",
                    f"--save-workers {int(args.teacher_cache_save_workers)}",
                    f"--num-shards {int(num_shards)}",
                    f"--shard-index {int(shard_index)}",
                    max_arg,
                    skip_arg,
                    "--override paths.cache_root=\"$CACHE_ROOT\"",
                    f"--override experiment={shq(args.experiment)}",
                    f"> \"$TEACHER_GAN_CACHE_ROOT/shard_{shard_index:05d}.log\" 2>&1",
                ]
            )
            + " &\n"
            + "pids+=($!)"
        )
    command_block = "\n".join(commands)
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
export CACHE_ROOT={shq(args.cache_root)}
export TEACHER_GAN_CACHE_ROOT={shq(args.teacher_cache_root)}
mkdir -p "$TEACHER_GAN_CACHE_ROOT"
pids=()
{command_block}
status=0
for pid in "${{pids[@]}}"; do
  wait "$pid" || status=1
done
exit "$status"
"""


def render_teacher_cache_manifest_check_script(args: argparse.Namespace, *, max_scenes: str) -> str:
    max_arg = f" --max-scenes {shq(max_scenes)}" if max_scenes else ""
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
export CACHE_ROOT={shq(args.cache_root)}
export TEACHER_GAN_CACHE_ROOT={shq(args.teacher_cache_root)}
PYTHONPATH=. python tools/build_self_forced_gan_teacher_cache.py \
  --ckpt-path {shq(args.pretrain_ckpt)} \
  --output-root "$TEACHER_GAN_CACHE_ROOT" \
  --split train \
  --rollouts-per-scene {int(args.teacher_cache_rollouts)} \
  --seed {int(args.teacher_cache_seed)} \
  --storage-dtype float16 \
  --amp-dtype {shq(args.teacher_cache_amp_dtype)} \
  --check-manifest{max_arg} \
  --batch-size {int(args.teacher_cache_batch_size)} \
  --rollout-batch-size {int(args.teacher_cache_rollout_batch_size)} \
  --data-num-workers {int(args.teacher_cache_data_num_workers)} \
  --data-prefetch-factor {int(args.teacher_cache_data_prefetch_factor)} \
  --save-workers {int(args.teacher_cache_save_workers)} \
  --override paths.cache_root="$CACHE_ROOT" \
  --override experiment={shq(args.experiment)}
PYTHONPATH=. python tools/validate_self_forced_gan_cache.py "$TEACHER_GAN_CACHE_ROOT" --max-files {int(args.validate_cache_max_files)}
"""


def render_merge_teacher_cache_index_script(args: argparse.Namespace, *, num_shards: int) -> str:
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
PYTHONPATH=. python tools/build_self_forced_gan_teacher_cache.py \
  --output-root {shq(args.teacher_cache_root)} \
  --merge-shard-indexes \
  --num-shards {int(num_shards)}
"""


def render_cache_validate_script(args: argparse.Namespace) -> str:
    return f"""set -Eeuo pipefail
cd {shq(args.project_root)}
if [[ -f /mnt/nuplan/miniforge/etc/profile.d/conda.sh ]]; then
  source /mnt/nuplan/miniforge/etc/profile.d/conda.sh
  conda activate catk
fi
PYTHONPATH=. python tools/validate_self_forced_gan_cache.py {shq(args.teacher_cache_root)} --max-files {int(args.validate_cache_max_files)}
"""


def sync_directory(args: argparse.Namespace, source_pod: str, dest_pod: str, directory: str) -> None:
    parent = str(Path(directory).parent)
    name = Path(directory).name
    if args.dry_run:
        print(
            " | ".join(
                [
                    f"kubectl exec -n {args.namespace} {source_pod} -c {args.container} -- tar czf - -C {shq(parent)} {shq(name)}",
                    f"kubectl exec -i -n {args.namespace} {dest_pod} -c {args.container} -- tar xzf - -C {shq(parent)}",
                ]
            )
        )
        return
    exec_in_pod(args, dest_pod, f"mkdir -p {shq(parent)}")
    source = subprocess.Popen(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            source_pod,
            "-c",
            args.container,
            "--",
            "tar",
            "czf",
            "-",
            "-C",
            parent,
            name,
        ],
        stdout=subprocess.PIPE,
    )
    assert source.stdout is not None
    dest = subprocess.run(
        [
            "kubectl",
            "exec",
            "-i",
            "-n",
            args.namespace,
            dest_pod,
            "-c",
            args.container,
            "--",
            "tar",
            "xzf",
            "-",
            "-C",
            parent,
        ],
        stdin=source.stdout,
        check=True,
    )
    source.stdout.close()
    source_status = source.wait()
    if source_status != 0:
        raise subprocess.CalledProcessError(source_status, source.args)
    if dest.returncode != 0:
        raise subprocess.CalledProcessError(dest.returncode, dest.args)


def sync_teacher_cache_shards(
    args: argparse.Namespace,
    source_pod: str,
    dest_pod: str,
    *,
    shard_start: int,
    shard_end: int,
    num_shards: int,
) -> None:
    root = args.teacher_cache_root
    list_script = f"""set -Eeuo pipefail
cd {shq(root)}
list_file="$(mktemp)"
python - "$list_file" <<'PY'
import json
import sys
from pathlib import Path

root = Path(".")
list_path = Path(sys.argv[1])
entries = []
num_shards = {int(num_shards)}
for shard_index in range({int(shard_start)}, {int(shard_end)}):
    index_name = f"index.shard_{{shard_index:05d}}_of_{{num_shards:05d}}.json"
    index_path = root / index_name
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    entries.append(index_name)
    manifest_name = f"teacher_cache_manifest.shard_{{shard_index:05d}}_of_{{num_shards:05d}}.json"
    if (root / manifest_name).exists():
        entries.append(manifest_name)
    log_name = f"shard_{{shard_index:05d}}.log"
    if (root / log_name).exists():
        entries.append(log_name)
    shard = json.loads(index_path.read_text())
    for value in shard.values():
        entries.append(str(value["path"] if isinstance(value, dict) and "path" in value else value))
list_path.write_text("\\n".join(dict.fromkeys(entries)) + "\\n")
PY
tar cf - -T "$list_file"
rm -f "$list_file"
"""
    if args.teacher_cache_direct_pod_sync:
        port = int(args.teacher_cache_sync_port)
        dest_ip = get_pod_ip(args, dest_pod)
        print(
            f"[launcher] direct shard sync {source_pod}->{dest_pod} "
            f"shards=[{shard_start},{shard_end}) port={port}",
            flush=True,
        )
        listener = popen_exec_in_pod(
            args,
            dest_pod,
            f"set -Eeuo pipefail; mkdir -p {shq(root)}; nc -l {port} | tar xf - -C {shq(root)}",
        )
        if not args.dry_run:
            import time as _time

            _time.sleep(2.0)
        source_script = f"({list_script}) | nc -N {shq(dest_ip)} {port}"
        try:
            exec_in_pod(args, source_pod, source_script)
        finally:
            return_code = listener.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, listener.args)
        return
    if args.dry_run:
        print(
            f"kubectl exec -n {args.namespace} {source_pod} -c {args.container} -- <shard-tar> "
            f"| kubectl exec -i -n {args.namespace} {dest_pod} -c {args.container} -- tar xf - -C {shq(root)}"
        )
        return
    exec_in_pod(args, dest_pod, f"mkdir -p {shq(root)}")
    source = subprocess.Popen(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            source_pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            list_script,
        ],
        stdout=subprocess.PIPE,
    )
    assert source.stdout is not None
    dest = subprocess.run(
        [
            "kubectl",
            "exec",
            "-i",
            "-n",
            args.namespace,
            dest_pod,
            "-c",
            args.container,
            "--",
            "tar",
            "xf",
            "-",
            "-C",
            root,
        ],
        stdin=source.stdout,
        check=True,
    )
    source.stdout.close()
    source_status = source.wait()
    if source_status != 0:
        raise subprocess.CalledProcessError(source_status, source.args)
    if dest.returncode != 0:
        raise subprocess.CalledProcessError(dest.returncode, dest.args)


def run_pod_scripts_parallel(
    args: argparse.Namespace,
    jobs: list[tuple[str, str]],
    *,
    label: str,
) -> None:
    if len(jobs) <= 1:
        for pod, script in jobs:
            exec_in_pod(args, pod, script)
        return
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(exec_in_pod, args, pod, script): pod
            for pod, script in jobs
        }
        for future in as_completed(futures):
            pod = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{label} failed on {pod}") from exc


def build_teacher_cache_parallel(args: argparse.Namespace) -> None:
    num_shards = len(args.pods) * int(args.teacher_cache_gpus_per_pod)
    started = time.monotonic()
    print(
        "[launcher] parallel teacher cache build "
        f"pods={len(args.pods)} gpus_per_pod={args.teacher_cache_gpus_per_pod} "
        f"num_shards={num_shards} batch_size={args.teacher_cache_batch_size} "
        f"rollout_batch_size={args.teacher_cache_rollout_batch_size}",
        flush=True,
    )
    processes = []
    for pod_index, pod in enumerate(args.pods):
        processes.append(
            (
                pod,
                popen_exec_in_pod(
                    args,
                    pod,
                    render_parallel_teacher_cache_build_script(
                        args,
                        pod_index=pod_index,
                        num_shards=num_shards,
                        max_scenes=args.teacher_cache_max_scenes,
                    ),
                ),
            )
        )
    failures = []
    for pod, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((pod, return_code))
    if failures:
        raise subprocess.CalledProcessError(
            failures[0][1],
            f"parallel teacher cache build failed: {failures}",
        )
    print(
        f"[launcher] teacher cache shard builders finished elapsed_sec={time.monotonic() - started:.1f}",
        flush=True,
    )

    if len(args.pods) > 1:
        if not args.sync_teacher_cache:
            raise ValueError("--parallel-teacher-cache with multiple pods requires --sync-teacher-cache.")
        shards_per_pod = int(args.teacher_cache_gpus_per_pod)
        print(f"[launcher] exchanging teacher cache shard files between {args.pods[0]} and {args.pods[1]}")
        sync_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    sync_teacher_cache_shards,
                    args,
                    args.pods[0],
                    args.pods[1],
                    shard_start=0,
                    shard_end=shards_per_pod,
                    num_shards=num_shards,
                ),
                executor.submit(
                    sync_teacher_cache_shards,
                    args,
                    args.pods[1],
                    args.pods[0],
                    shard_start=shards_per_pod,
                    shard_end=num_shards,
                    num_shards=num_shards,
                ),
            ]
            for future in as_completed(futures):
                future.result()
        print(
            f"[launcher] teacher cache shard exchange finished elapsed_sec={time.monotonic() - sync_started:.1f}",
            flush=True,
        )

    print("[launcher] merging shard indexes on all cache pods")
    merge_started = time.monotonic()
    run_pod_scripts_parallel(
        args,
        [
            (pod, render_merge_teacher_cache_index_script(args, num_shards=num_shards))
            for pod in args.pods
        ],
        label="teacher cache index merge",
    )
    print(
        f"[launcher] teacher cache index merge finished elapsed_sec={time.monotonic() - merge_started:.1f}",
        flush=True,
    )


def matching_teacher_cache_available(args: argparse.Namespace) -> bool:
    outputs = {}
    with ThreadPoolExecutor(max_workers=len(args.pods)) as executor:
        futures = {
            executor.submit(
                exec_in_pod,
                args,
                pod,
                render_teacher_cache_manifest_check_script(args, max_scenes=args.teacher_cache_max_scenes),
                capture=True,
            ): pod
            for pod in args.pods
        }
        for future in as_completed(futures):
            pod = futures[future]
            try:
                outputs[pod] = future.result()
            except subprocess.CalledProcessError:
                print(f"[launcher] no matching teacher cache manifest on {pod}", flush=True)
                return False
    for pod, output in outputs.items():
        if output:
            print(f"[launcher] matching teacher cache on {pod}:\n{output}", flush=True)
    print("[launcher] matching teacher cache is already available on all pods; skipping build", flush=True)
    return True


def base_launcher_command(args: argparse.Namespace) -> list[str]:
    effective_scene_batch = (
        int(args.train_batch_size)
        * int(args.nproc_per_node)
        * len(args.pods)
        * int(args.accumulate_grad_batches)
    )
    extra_overrides = [
        f"paths.teacher_gan_cache_root={args.teacher_cache_root}",
        f"trainer.precision={args.precision}",
        "trainer.accumulate_grad_batches=1",
        f"model.model_config.self_forced_gan.manual_accumulate_grad_batches={int(args.accumulate_grad_batches)}",
        f"model.model_config.self_forced_gan.effective_scene_batch={effective_scene_batch}",
    ]
    if args.disable_validation:
        extra_overrides.extend(
            [
                "trainer.limit_val_batches=0",
                "model.model_config.val_open_loop=false",
                "model.model_config.val_closed_loop=false",
                "model.model_config.scorer_scene_num=0",
            ]
        )
    if args.extra_hydra_overrides:
        extra_overrides.extend(shlex.split(args.extra_hydra_overrides))

    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pods",
        *args.pods,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--cache-root",
        args.cache_root,
        "--action",
        "finetune",
        "--ckpt-path",
        args.pretrain_ckpt,
        "--experiment",
        args.experiment,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--master-port",
        args.master_port,
        "--checkpoint-sync-port",
        args.checkpoint_sync_port,
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--log-dir",
        args.log_dir,
        "--train-batch-size",
        str(args.train_batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--extra-hydra-overrides",
        " ".join(extra_overrides),
        "--remote-env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "--remote-env",
        "NCCL_ALGO=Ring",
        "--remote-env",
        "NCCL_PROTO=Simple",
        "--remote-env",
        "CATK_ATTENTION_GRAPH_FP32=1",
    ]
    if args.limit_train_batches:
        command.extend(["--limit-train-batches", str(args.limit_train_batches)])
    if args.limit_val_batches:
        command.extend(["--limit-val-batches", str(args.limit_val_batches)])
    if args.no_pull:
        command.append("--no-pull")
    if args.git_ref:
        command.extend(["--git-ref", args.git_ref])
    if args.no_monitor_pane:
        command.append("--no-monitor-pane")
    if args.replace:
        command.append("--replace")
    if args.dry_run:
        command.append("--dry-run")
    return command


def stop_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BASE_LAUNCHER),
        "--namespace",
        args.namespace,
        "--container",
        args.container,
        "--pods",
        *args.pods,
        "--project-root",
        args.project_root,
        "--branch",
        args.branch,
        "--task-name",
        args.task_name,
        "--session",
        args.session,
        "--stop",
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DEFAULT_DESCRIPTION,
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--pods", nargs="+", default=list(DEFAULT_PODS))
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--git-ref", default="")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--teacher-cache-root", default=DEFAULT_TEACHER_CACHE_ROOT)
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument("--pretrain-download-dir", default=DEFAULT_PRETRAIN_DOWNLOAD_DIR)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-pretrain-task", default=DEFAULT_WANDB_PRETRAIN_TASK)
    parser.add_argument(
        "--wandb-pretrain-artifact",
        default=DEFAULT_WANDB_PRETRAIN_ARTIFACT,
        help=(
            "Exact model artifact ref. Defaults to the pinned epoch-last-sqverrgj:v38. "
            "Pass an empty string to resolve the latest epoch-last artifact from --wandb-pretrain-task."
        ),
    )
    parser.add_argument("--force-pretrain-download", action="store_true")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--master-port", default=DEFAULT_MASTER_PORT)
    parser.add_argument("--checkpoint-sync-port", default=DEFAULT_CHECKPOINT_SYNC_PORT)
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--accumulate-grad-batches", type=int, default=DEFAULT_ACCUMULATE_GRAD_BATCHES)
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument("--precision", default=DEFAULT_PRECISION)
    parser.add_argument("--max-epochs", default="6")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--build-teacher-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument(
        "--teacher-cache-max-scenes",
        default="",
        help="Debug/smoke limit for cache building. Omit when building the full train cache.",
    )
    parser.add_argument("--teacher-cache-rollouts", type=int, default=32)
    parser.add_argument("--teacher-cache-seed", type=int, default=817)
    parser.add_argument("--teacher-cache-batch-size", type=int, default=DEFAULT_TEACHER_CACHE_BATCH_SIZE)
    parser.add_argument("--teacher-cache-rollout-batch-size", type=int, default=DEFAULT_TEACHER_CACHE_ROLLOUT_BATCH_SIZE)
    parser.add_argument("--teacher-cache-data-num-workers", type=int, default=DEFAULT_TEACHER_CACHE_DATA_NUM_WORKERS)
    parser.add_argument(
        "--teacher-cache-data-prefetch-factor",
        type=int,
        default=DEFAULT_TEACHER_CACHE_DATA_PREFETCH_FACTOR,
    )
    parser.add_argument("--teacher-cache-save-workers", type=int, default=DEFAULT_TEACHER_CACHE_SAVE_WORKERS)
    parser.add_argument("--teacher-cache-gpus-per-pod", type=int, default=DEFAULT_TEACHER_CACHE_GPUS_PER_POD)
    parser.add_argument(
        "--teacher-cache-amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default=DEFAULT_TEACHER_CACHE_AMP_DTYPE,
    )
    parser.add_argument("--teacher-cache-sync-port", default="29720")
    parser.add_argument("--teacher-cache-direct-pod-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-teacher-cache", action="store_true")
    parser.add_argument("--teacher-cache-keyed-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-matching-teacher-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-teacher-cache", action="store_true")
    parser.add_argument("--sync-teacher-cache", action="store_true")
    parser.add_argument("--validate-cache-max-files", type=int, default=16)
    parser.add_argument("--skip-pretrain-download", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    args = parser.parse_args()
    if len(args.pods) != DEFAULT_EXPECTED_POD_COUNT and not args.stop:
        parser.error(f"this preset expects exactly {DEFAULT_EXPECTED_POD_COUNT} pod(s)")
    if args.nproc_per_node != DEFAULT_NPROC_PER_NODE and not args.stop:
        parser.error(f"--nproc-per-node must be {DEFAULT_NPROC_PER_NODE} for this preset")
    if args.teacher_cache_rollouts < 32:
        parser.error("--teacher-cache-rollouts must be at least 32 for K=16 sampling from cache[32]")
    if args.teacher_cache_seed < 0:
        parser.error("--teacher-cache-seed must be >= 0")
    if args.teacher_cache_batch_size < 1:
        parser.error("--teacher-cache-batch-size must be >= 1")
    if args.teacher_cache_rollout_batch_size < 1:
        parser.error("--teacher-cache-rollout-batch-size must be >= 1")
    if args.teacher_cache_data_num_workers < 0:
        parser.error("--teacher-cache-data-num-workers must be >= 0")
    if args.teacher_cache_data_prefetch_factor < 1:
        parser.error("--teacher-cache-data-prefetch-factor must be >= 1")
    if args.teacher_cache_save_workers < 0:
        parser.error("--teacher-cache-save-workers must be >= 0")
    if args.teacher_cache_gpus_per_pod < 1:
        parser.error("--teacher-cache-gpus-per-pod must be >= 1")
    if args.sync_teacher_cache and len(args.pods) < 2 and not args.stop:
        parser.error("--sync-teacher-cache requires at least two pods")
    return args


def main() -> int:
    args = parse_args()
    if args.stop:
        return subprocess.call(stop_command(args))

    if not args.skip_pretrain_download:
        target_pods = args.pods if args.parallel_teacher_cache else [args.pods[0]]
        mode = "pinned" if args.wandb_pretrain_artifact else "latest"
        print(f"[launcher] preparing {mode} W&B pretrain checkpoint on {', '.join(target_pods)}")
        run_pod_scripts_parallel(
            args,
            [(pod, render_pretrain_download_script(args)) for pod in target_pods],
            label="pretrain checkpoint preparation",
        )

    resolve_teacher_cache_root(args)

    if args.build_teacher_cache:
        cache_ready = False
        if args.reuse_matching_teacher_cache:
            cache_ready = matching_teacher_cache_available(args)
        if not cache_ready:
            if args.parallel_teacher_cache:
                build_teacher_cache_parallel(args)
            else:
                print(f"[launcher] building teacher cache on {args.pods[0]}: {args.teacher_cache_root}")
                exec_in_pod(
                    args,
                    args.pods[0],
                    render_teacher_cache_build_script(args, max_scenes=args.teacher_cache_max_scenes),
                )
                if args.sync_teacher_cache:
                    print(f"[launcher] syncing teacher cache to {args.pods[1]}")
                    sync_directory(args, args.pods[0], args.pods[1], args.teacher_cache_root)

    print("[launcher] validating teacher cache on configured pod(s)")
    validate_started = time.monotonic()
    run_pod_scripts_parallel(
        args,
        [(pod, render_cache_validate_script(args)) for pod in args.pods],
        label="teacher cache validation",
    )
    print(
        f"[launcher] teacher cache validation finished elapsed_sec={time.monotonic() - validate_started:.1f}",
        flush=True,
    )
    if args.build_cache_only:
        print("[launcher] build-cache-only requested; skipping multi-node training launch")
        return 0

    command = base_launcher_command(args)
    print("[launcher] starting multi-node GAN fine-tuning")
    print(" ".join(shq(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
