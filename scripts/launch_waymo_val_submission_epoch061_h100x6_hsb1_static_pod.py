#!/usr/bin/env python3
"""Launch a guarded H100x6 Waymo validation submission export on hsb-npc-training-1.

The default mode is safe: it generates and verifies a validation-set submission
archive but does not upload to the Waymo validation leaderboard. Passing
``--submit-validation`` explicitly arms the final upload.

The launcher only uses an existing pod. It never creates, deletes, or restarts
pods.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import os
import shlex
import subprocess
import textwrap


DEFAULT_NAMESPACE = "p-pnc"
DEFAULT_CONTAINER = "main"
DEFAULT_POD = "hsb-npc-training-1"
DEFAULT_BRANCH = "semi_control_stable"
DEFAULT_PROJECT_ROOT = "/mnt/nuplan/projects/catk"
DEFAULT_CACHE_ROOT = "/workspace/womd_v1_3/SMART_cache"
DEFAULT_LOG_DIR = "/mnt/nuplan/projects/catk/logs"
DEFAULT_ARTIFACT = "jksg01019-naver-labs/SMART-FLOW/epoch-last-x5f9g0ce:v57"
DEFAULT_EPOCH = 61
DEFAULT_NOISE_SCALE = "1.0"
DEFAULT_ANTITHETIC_PAIRS = "true"
DEFAULT_STRATIFIED_GAUSSIAN_NOISE = "false"
DESCRIPTION_PREFIX = (
    "flow_control_space_pretrain_h100x4_h100x2_prefix_default_noslip_"
    "tailprefix_roundtrip05_lr6e-4_bs20_epoch061"
)
DEFAULT_VAL_BATCH_SIZE = 48
DEFAULT_SMOKE_VAL_BATCH_SIZE = 8


def shq(value: object) -> str:
    return shlex.quote(str(value))


def strip_template_indent(text: str) -> str:
    """Remove indentation introduced by nested Python string templates."""

    return textwrap.dedent(text).strip("\n") + "\n"


def validate_noise_scale(value: str) -> str:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid noise scale: {value}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("noise scale must be positive")
    return value


def noise_tag(value: str) -> str:
    parsed = Decimal(value)
    return f"{int((parsed * Decimal(1000)).to_integral_value()):04d}"


def pair_label(value: str) -> str:
    return "antithetic" if value == "true" else "iid"


def stratified_label(value: str) -> str:
    return "stratified" if value == "true" else "iidgaussian"


def run_kubectl(args: list[str], *, capture: bool = False, dry_run: bool = False) -> str:
    command = ["kubectl", *args]
    if dry_run:
        print("+ " + " ".join(shq(part) for part in command))
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return result.stdout.strip() if capture and result.stdout is not None else ""


def render_remote_script(args: argparse.Namespace) -> str:
    # `--verify-waymo-ui` must arm the validation submission code path so that
    # the Waymo validation upload form is located, but `dry_run=true` returns before
    # attaching the archive or clicking submit.
    submit_validate = "true" if (args.submit_validation or args.verify_waymo_ui) else "false"
    waymo_enabled = "true" if (args.submit_validation or args.verify_waymo_ui) else "false"
    waymo_dry_run = "true" if args.verify_waymo_ui and not args.submit_validation else "false"
    limit_val_batches = "1" if args.smoke_test else str(args.limit_val_batches)
    val_batch_size = (
        args.smoke_val_batch_size if args.smoke_test else args.val_batch_size
    )
    expected_validation_scenarios = (
        ""
        if args.smoke_test or args.expected_validation_scenarios is None
        else str(int(args.expected_validation_scenarios))
    )

    hydra_overrides = [
        "experiment=sim_agents_sub_flow",
        "action=validate",
        "trainer=ddp",
        "trainer.devices=6",
        "trainer.num_nodes=1",
        "trainer.limit_val_batches=${LIMIT_VAL_BATCHES}",
        "data.val_batch_size=${VAL_BATCH_SIZE}",
        "data.test_batch_size=${VAL_BATCH_SIZE}",
        "paths.cache_root=${CACHE_ROOT}",
        "paths.log_dir=${LOG_DIR}",
        "task_name=${TASK_NAME}",
        "hydra.run.dir=${RUN_DIR}",
        "ckpt_path=${CKPT_PATH}",
        "logger.wandb.group=${WANDB_GROUP}",
        "logger.wandb.job_type=waymo_validation_submission",
        "logger.wandb.log_model=false",
        "model.model_config.val_open_loop=false",
        "model.model_config.val_closed_loop=true",
        "model.model_config.n_batch_sim_agents_metric=0",
        "model.model_config.scorer_scene_num=0",
        "model.model_config.n_rollout_closed_val=32",
        "model.model_config.validation_closed_seed=4",
        "model.model_config.validation_rollout_sampling.sample_steps=16",
        "model.model_config.validation_rollout_sampling.sample_method=euler",
        f"model.model_config.validation_rollout_sampling.noise_scale={args.noise_scale}",
        f"model.model_config.validation_rollout_sampling.antithetic_pairs={args.antithetic_pairs}",
        f"model.model_config.validation_rollout_sampling.stratified_gaussian_noise={args.stratified_gaussian_noise}",
        "model.model_config.decoder.flow_solver_method=euler",
        "model.model_config.decoder.use_lqr=false",
        "model.model_config.decoder.use_stop_motion=false",
        "model.model_config.self_forced.use_stop_motion=false",
        'model.model_config.sim_agents_submission.method_name="Flow Agents 7M"',
        'model.model_config.sim_agents_submission.authors=["SB H","KO O"]',
        "model.model_config.sim_agents_submission.affiliation=NLK",
        f'model.model_config.sim_agents_submission.description="{args.description}"',
        'model.model_config.sim_agents_submission.method_link="not available yet"',
        "model.model_config.sim_agents_submission.account_name=h.sb@naverlabs.com",
        f"waymo_submission.enabled={waymo_enabled}",
        f"waymo_submission.submit_validate={submit_validate}",
        "waymo_submission.submit_test=false",
        "waymo_submission.evaluation_set=validation",
        f"waymo_submission.dry_run={waymo_dry_run}",
        "waymo_submission.poll_submission_status=false",
        f"waymo_submission.upload_timeout_ms={int(args.upload_timeout_ms)}",
    ]
    if args.storage_state_path:
        hydra_overrides.append(f"waymo_submission.storage_state_path={args.storage_state_path}")
    if args.extra_hydra_overrides:
        hydra_overrides.extend(args.extra_hydra_overrides)

    hydra_array = "\n".join(hydra_overrides)
    hydra_marker = "__CATK_HYDRA_OVERRIDES__"
    download_python_marker = "__CATK_DOWNLOAD_CHECKPOINT_PY__"
    download_python = textwrap.dedent(
        """\
        import sys
        from pathlib import Path
        import wandb

        artifact_name, destination = sys.argv[1], Path(sys.argv[2])
        destination.parent.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project="SMART-FLOW",
            entity="jksg01019-naver-labs",
            job_type="download_waymo_validation_checkpoint",
            mode="online",
        )
        artifact = run.use_artifact(artifact_name, type="model")
        root = Path(artifact.download(root=str(destination.parent / "artifact")))
        candidates = sorted(root.rglob("*.ckpt"))
        if not candidates:
            raise SystemExit(f"no .ckpt found in {artifact_name}")
        destination.write_bytes(candidates[0].read_bytes())
        print(destination)
        run.finish()
        """
    ).rstrip()
    verify_command = (
        "python scripts/verify_waymo_submission_archive.py "
        "--archive \"$ARCHIVE_PATH\" "
        "--expected-rollouts-per-scenario 32 "
        "--expected-steps-per-trajectory 80 "
        "--expected-method-name \"Flow Agents 7M\" "
        "--expected-authors \"SB H,KO O\" "
        "--expected-affiliation NLK "
        f"--expected-description {shq(args.description)} "
        "--expected-method-link \"not available yet\" "
        "--expected-account-name h.sb@naverlabs.com "
        "--expected-num-model-parameters 7M "
        "--require-closed-loop-ack"
    )

    script = strip_template_indent(
        f"""\
        #!/usr/bin/env bash
        set -Eeuo pipefail

        export TERM="${{TERM:-xterm-256color}}"
        export HYDRA_FULL_ERROR=1
        export TF_CPP_MIN_LOG_LEVEL=2
        export PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
        export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
        export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
        export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
        export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
        export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-eth0}}"
        export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-eth0}}"
        export NCCL_SOCKET_FAMILY="${{NCCL_SOCKET_FAMILY:-AF_INET}}"
        export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
        export NCCL_NVLS_ENABLE="${{NCCL_NVLS_ENABLE:-0}}"
        export NCCL_CUMEM_ENABLE="${{NCCL_CUMEM_ENABLE:-0}}"
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${{TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}}"
        export WANDB_ENTITY="${{WANDB_ENTITY:-jksg01019-naver-labs}}"
        export WANDB_PROJECT="${{WANDB_PROJECT:-SMART-FLOW}}"
        export CATK_REMOTE_PYTHON="${{CATK_REMOTE_PYTHON:-/mnt/nuplan/miniforge/envs/catk/bin/python}}"

        PROJECT_ROOT={shq(args.project_root)}
        CACHE_ROOT={shq(args.cache_root)}
        LOG_DIR={shq(args.log_dir)}
        TASK_NAME={shq(args.task_name)}
        RUN_ID={shq(args.run_id)}
        RUN_DIR="${{LOG_DIR%/}}/${{TASK_NAME}}/runs/${{RUN_ID}}"
        CHECKPOINT_ARTIFACT={shq(args.artifact)}
        CKPT_DIR="${{LOG_DIR%/}}/${{TASK_NAME}}/checkpoints/epoch_{int(args.epoch):03d}"
        CKPT_PATH="${{CKPT_DIR}}/epoch_last.ckpt"
        VAL_BATCH_SIZE={int(val_batch_size)}
        LIMIT_VAL_BATCHES={shq(limit_val_batches)}
        WANDB_GROUP={shq(args.wandb_group)}
        ARCHIVE_PATH="${{RUN_DIR}}/sim_agents_2025_submission.tar.gz"
        STATUS_FILE="${{LOG_DIR%/}}/${{TASK_NAME}}/${{RUN_ID}}.status.log"
        EXPECTED_VALIDATION_SCENARIOS={shq(expected_validation_scenarios)}

        cd "$PROJECT_ROOT"
        mkdir -p "$RUN_DIR" "$CKPT_DIR" "$(dirname "$STATUS_FILE")"

        echo "[$(date '+%F %T')] start waymo validation submission export"
        echo "  branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse --short HEAD)"
        echo "  artifact=$CHECKPOINT_ARTIFACT"
        echo "  run_dir=$RUN_DIR"
        echo "  submit_validate={submit_validate} submit_test=false waymo_enabled={waymo_enabled} waymo_dry_run={waymo_dry_run}"
        echo "  smoke_test={str(args.smoke_test).lower()} limit_val_batches=$LIMIT_VAL_BATCHES val_batch_size=$VAL_BATCH_SIZE"

        if [[ ! -d "$CACHE_ROOT/validation" ]]; then
          echo "ERROR: missing validation cache directory: $CACHE_ROOT/validation" >&2
          exit 2
        fi
        if [[ "{str(args.smoke_test).lower()}" != "true" && -z "$EXPECTED_VALIDATION_SCENARIOS" ]]; then
          EXPECTED_VALIDATION_SCENARIOS=$(find "$CACHE_ROOT/validation" -type f -name '*.pkl' | wc -l | tr -d ' ')
          echo "  auto expected_validation_scenarios=$EXPECTED_VALIDATION_SCENARIOS"
        fi

        if [[ ! -f "$CKPT_PATH" ]]; then
          echo "[$(date '+%F %T')] downloading checkpoint artifact $CHECKPOINT_ARTIFACT"
          "$CATK_REMOTE_PYTHON" - "$CHECKPOINT_ARTIFACT" "$CKPT_PATH" <<'PY'
        {download_python_marker}
        PY
        fi

        test -f "$CKPT_PATH"
        sha256sum "$CKPT_PATH" | tee "$CKPT_PATH.sha256"

        mapfile -t HYDRA_OVERRIDES <<CATK_OVERRIDES
        {hydra_marker}
        CATK_OVERRIDES

        echo "[$(date '+%F %T')] torchrun start"
        printf '  %q' torchrun --standalone --nproc_per_node=6 -m src.run "${{HYDRA_OVERRIDES[@]}}"
        printf '\\n'
        torchrun --standalone --nproc_per_node=6 -m src.run "${{HYDRA_OVERRIDES[@]}}"

        echo "[$(date '+%F %T')] verifying archive $ARCHIVE_PATH"
        VERIFY_SCENARIOS_ARG=()
        if [[ -n "$EXPECTED_VALIDATION_SCENARIOS" ]]; then
          VERIFY_SCENARIOS_ARG=(--expected-scenarios "$EXPECTED_VALIDATION_SCENARIOS")
        fi
        {verify_command} "${{VERIFY_SCENARIOS_ARG[@]}}"

        echo "[$(date '+%F %T')] DONE archive=$ARCHIVE_PATH"
        """
    )
    return script.replace(download_python_marker, download_python).replace(
        hydra_marker,
        hydra_array,
    )


def render_start_command(args: argparse.Namespace) -> str:
    run_root = f"{args.log_dir.rstrip('/')}/{args.task_name}/launcher"
    run_file = f"{run_root}/{args.run_id}_run.sh"
    log_file = f"{run_root}/{args.run_id}.tmux.log"
    remote_script = render_remote_script(args)
    remote_script_marker = "__CATK_REMOTE_SCRIPT_CONTENT__"
    pull_block = ""
    if not args.no_pull:
        pull_block = textwrap.dedent(
            f"""\
            git config --global --add safe.directory {shq(args.project_root)} || true
            git update-ref -d {shq(f"refs/remotes/origin/{args.branch}")} || true
            git fetch origin --prune {shq(f"+{args.branch}:refs/remotes/origin/{args.branch}")}
            git checkout -f {shq(f"origin/{args.branch}")}
            """
        )
    replace_block = ""
    if args.replace:
        replace_block = (
            f"tmux has-session -t {shq(args.session)} 2>/dev/null "
            f"&& tmux kill-session -t {shq(args.session)} || true"
        )
    else:
        replace_block = textwrap.dedent(
            f"""\
            if tmux has-session -t {shq(args.session)} 2>/dev/null; then
              echo "tmux session already exists: {args.session}" >&2
              exit 3
            fi
            """
        )
    # Keep the here-doc delimiter at column 0. If it is indented, the remote
    # shell treats the launcher tail as part of the generated run script.
    script = f"""set -Eeuo pipefail
cd {shq(args.project_root)}
{pull_block}
{replace_block}
mkdir -p {shq(run_root)}
cat > {shq(run_file)} <<'CATK_REMOTE_SCRIPT'
{remote_script_marker}
CATK_REMOTE_SCRIPT
chmod +x {shq(run_file)}
: > {shq(log_file)}
tmux new-session -d -s {shq(args.session)} -c {shq(args.project_root)} {shq(run_file)}
tmux pipe-pane -t {shq(args.session)} -o {shq("cat >> " + log_file)}
echo "[launcher] started tmux session {args.session} on {args.pod}"
echo "[launcher] tmux log: {log_file}"
echo "[launcher] run dir: {args.log_dir.rstrip('/')}/{args.task_name}/runs/{args.run_id}"
"""
    return script.replace(remote_script_marker, remote_script.rstrip())


def render_stop_command(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f"""\
        set -Eeuo pipefail
        if tmux has-session -t {shq(args.session)} 2>/dev/null; then
          tmux kill-session -t {shq(args.session)}
          echo "[launcher] stopped tmux session {args.session}"
        else
          echo "[launcher] tmux session not found: {args.session}"
        fi
        mapfile -t pids < <(pgrep -f "task_name={args.task_name}" 2>/dev/null || true)
        if (( ${{#pids[@]}} > 0 )); then
          echo "[launcher] terminating task processes: ${{pids[*]}}"
          kill -TERM "${{pids[@]}}" 2>/dev/null || true
          sleep 10
          mapfile -t pids < <(pgrep -f "task_name={args.task_name}" 2>/dev/null || true)
          if (( ${{#pids[@]}} > 0 )); then
            kill -KILL "${{pids[@]}}" 2>/dev/null || true
          fi
        fi
        """
    )


def exec_in_pod(args: argparse.Namespace, script: str) -> None:
    kubectl_args = [
        "exec",
        "-n",
        args.namespace,
        args.pod,
        "-c",
        args.container,
        "--",
        "bash",
        "-lc",
        script,
    ]
    run_kubectl(kubectl_args, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--noise-scale", type=validate_noise_scale, default=DEFAULT_NOISE_SCALE)
    parser.add_argument(
        "--antithetic-pairs",
        choices=["true", "false"],
        default=DEFAULT_ANTITHETIC_PAIRS,
    )
    parser.add_argument(
        "--stratified-gaussian-noise",
        choices=["true", "false"],
        default=DEFAULT_STRATIFIED_GAUSSIAN_NOISE,
        help=(
            "Use coordinate-wise stratified Gaussian base noise in the scenario-seeded "
            "closed-loop rollout path. Intended to be used with --antithetic-pairs true."
        ),
    )
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--run-id",
        default=os.environ.get("CATK_RUN_ID", ""),
        help="Fixed Hydra run id. Defaults to a timestamp if omitted.",
    )
    parser.add_argument("--session", default="")
    parser.add_argument("--val-batch-size", type=int, default=DEFAULT_VAL_BATCH_SIZE)
    parser.add_argument(
        "--smoke-val-batch-size",
        type=int,
        default=DEFAULT_SMOKE_VAL_BATCH_SIZE,
    )
    parser.add_argument("--limit-val-batches", default="1.0")
    parser.add_argument("--expected-validation-scenarios", type=int, default=None)
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--storage-state-path", default="")
    parser.add_argument("--upload-timeout-ms", type=int, default=7200000)
    parser.add_argument("--extra-hydra-overrides", nargs="*", default=[])
    parser.add_argument("--submit-validation", action="store_true")
    parser.add_argument(
        "--verify-waymo-ui",
        action="store_true",
        help=(
            "After archive generation, open the Waymo validation submission UI and "
            "verify the upload form, but do not attach or submit the archive."
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    args = parser.parse_args()

    if not args.run_id:
        import datetime as dt

        suffix = "smoke" if args.smoke_test else "full"
        args.run_id = dt.datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")
    if not args.task_name:
        args.task_name = (
            "flow_agents_7m_waymo_val_epoch061_x5f9g0ce_h100x6_hsb1_"
            f"sample16_euler_{pair_label(args.antithetic_pairs)}_"
            f"{stratified_label(args.stratified_gaussian_noise)}_"
            f"noise{noise_tag(args.noise_scale)}"
        )
    if not args.description:
        args.description = (
            f"{DESCRIPTION_PREFIX}_{args.antithetic_pairs}_"
            f"stratified_{args.stratified_gaussian_noise}_{args.noise_scale}"
        )
    if not args.session:
        args.session = (
            "catk-flow-waymo-val-submission-epoch061-h100x6-hsb1-"
            f"{pair_label(args.antithetic_pairs)}-"
            f"{stratified_label(args.stratified_gaussian_noise)}-"
            f"noise{noise_tag(args.noise_scale)}"
        )
    if not args.wandb_group:
        args.wandb_group = f"{args.task_name}_submission_export"
    if args.submit_validation and args.smoke_test:
        parser.error("--submit-validation is not allowed together with --smoke-test")
    if args.submit_validation and args.verify_waymo_ui:
        parser.error("--submit-validation and --verify-waymo-ui are mutually exclusive")
    if args.val_batch_size < 1 or args.smoke_val_batch_size < 1:
        parser.error("batch sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.stop:
        exec_in_pod(args, render_stop_command(args))
    else:
        exec_in_pod(args, render_start_command(args))
        print(
            "attach: kubectl exec -it "
            f"-n {args.namespace} {args.pod} -c {args.container} -- "
            f"tmux attach -t {args.session}"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
