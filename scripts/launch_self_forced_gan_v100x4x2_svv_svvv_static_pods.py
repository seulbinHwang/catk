#!/usr/bin/env python3
"""Launch Set-level Self-Forced GAN fine-tuning on svv + svvv.

This launcher targets two already-running static V100x4 pods. It never creates,
deletes, or restarts pods. It prepares the W&B pretrain checkpoint on rank 0,
optionally builds/syncs the offline teacher rollout cache, then delegates the
actual tmux multi-node run to ``launch_h100x4_multinode_pretrain_tmux.py``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
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
DEFAULT_WANDB_ENTITY = "jksg01019-naver-labs"
DEFAULT_WANDB_PROJECT = "SMART-FLOW"
DEFAULT_WANDB_PRETRAIN_TASK = (
    "flow_control_space_pretrain_h100x6_hsb2_wo1_execctx_prefix_balanced_"
    "lr6e-4_bs18_oomretry"
)
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
        print(
            "[pretrain] using cached latest W&B checkpoint "
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


def render_teacher_cache_build_script(args: argparse.Namespace, *, max_scenes: str) -> str:
    max_arg = f" --max-scenes {shq(max_scenes)}" if max_scenes else ""
    skip_arg = " --skip-existing" if args.skip_existing_teacher_cache else ""
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
  --storage-dtype float16 \
  --amp-dtype float16 \
  --device cuda:0{max_arg}{skip_arg} \
  --override paths.cache_root="$CACHE_ROOT" \
  --override experiment={shq(args.experiment)}
PYTHONPATH=. python tools/validate_self_forced_gan_cache.py "$TEACHER_GAN_CACHE_ROOT" --max-files {int(args.validate_cache_max_files)}
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


def base_launcher_command(args: argparse.Namespace) -> list[str]:
    extra_overrides = [
        f"paths.teacher_gan_cache_root={args.teacher_cache_root}",
        f"trainer.precision={args.precision}",
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
        description="Launch Set-level Self-Forced GAN fine-tuning on svv + svvv V100x4 pods.",
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
        default="",
        help="Optional exact artifact ref. If omitted, the latest epoch-last artifact from --wandb-pretrain-task is used.",
    )
    parser.add_argument("--force-pretrain-download", action="store_true")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--master-port", default="29670")
    parser.add_argument("--checkpoint-sync-port", default="29671")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--max-epochs", default="6")
    parser.add_argument("--limit-train-batches", default="")
    parser.add_argument("--limit-val-batches", default="")
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--extra-hydra-overrides", default="")
    parser.add_argument("--build-teacher-cache", action="store_true")
    parser.add_argument(
        "--teacher-cache-max-scenes",
        default="",
        help="Debug/smoke limit for cache building. Omit when building the full train cache.",
    )
    parser.add_argument("--teacher-cache-rollouts", type=int, default=32)
    parser.add_argument("--skip-existing-teacher-cache", action="store_true")
    parser.add_argument("--sync-teacher-cache", action="store_true")
    parser.add_argument("--validate-cache-max-files", type=int, default=16)
    parser.add_argument("--skip-pretrain-download", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-monitor-pane", action="store_true")
    args = parser.parse_args()
    if len(args.pods) != 2 and not args.stop:
        parser.error("this preset expects exactly two pods")
    if args.nproc_per_node != 4 and not args.stop:
        parser.error("--nproc-per-node must be 4 for svv/svvv V100x4 pods")
    if args.teacher_cache_rollouts < 32:
        parser.error("--teacher-cache-rollouts must be at least 32 for K=16 sampling from cache[32]")
    return args


def main() -> int:
    args = parse_args()
    if args.stop:
        return subprocess.call(stop_command(args))

    if not args.skip_pretrain_download:
        print(f"[launcher] preparing latest W&B pretrain checkpoint on {args.pods[0]}")
        exec_in_pod(args, args.pods[0], render_pretrain_download_script(args))

    if args.build_teacher_cache:
        print(f"[launcher] building teacher cache on {args.pods[0]}: {args.teacher_cache_root}")
        exec_in_pod(
            args,
            args.pods[0],
            render_teacher_cache_build_script(args, max_scenes=args.teacher_cache_max_scenes),
        )
        if args.sync_teacher_cache:
            print(f"[launcher] syncing teacher cache to {args.pods[1]}")
            sync_directory(args, args.pods[0], args.pods[1], args.teacher_cache_root)

    print("[launcher] validating teacher cache on both pods")
    for pod in args.pods:
        exec_in_pod(args, pod, render_cache_validate_script(args))

    command = base_launcher_command(args)
    print("[launcher] starting multi-node GAN fine-tuning")
    print(" ".join(shq(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
