#!/usr/bin/env python3
"""Autonomous CLSFT experiment supervisor for the testsv pod.

This script is intentionally conservative.  It can stop clearly bad runs and
launch the next configured hyperparameter candidate from the warmup checkpoint.
It does not synthesize code patches by itself; code-level failures are logged and
left stopped for a human/Codex intervention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


RMM_KEY = "val_closed/sim_agents_2025/realism_meta_metric"
CPD_KEY = "val_closed/WOSAC-CPD/value"
ESTIMATOR_LOSS_KEY = "train/sf_generated_estimator_loss_step"
TRAIN_LOSS_KEY = "train/loss_step"
GLOBAL_STEP_KEY = "trainer/global_step"


@dataclass(frozen=True)
class Candidate:
    name: str
    lr: str = "1.0e-7"
    estimator_lr: str = "1.0e-7"
    beta: str = "1.0"
    use_anchor: str = "false"
    anchor_weight: str = "0.1"
    train_b: str = "4"
    val_b: str = "4"
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)


CANDIDATES = [
    Candidate(
        name="lr5e8_beta1",
        lr="5.0e-8",
        estimator_lr="5.0e-8",
        beta="1.0",
        notes="RMM low: reduce generator and estimator LR.",
    ),
    Candidate(
        name="lr2e8_beta1",
        lr="2.0e-8",
        estimator_lr="2.0e-8",
        beta="1.0",
        notes="RMM low: reduce LR further while keeping beta fixed.",
    ),
    Candidate(
        name="lr1e8_beta1",
        lr="1.0e-8",
        estimator_lr="1.0e-8",
        beta="1.0",
        notes="RMM low: final LR-only conservative attempt.",
    ),
    Candidate(
        name="lr1e7_beta1_anchor005",
        lr="1.0e-7",
        estimator_lr="1.0e-7",
        beta="1.0",
        use_anchor="true",
        anchor_weight="0.05",
        notes="Keep beta fixed and add light anchor FM regularization.",
    ),
    Candidate(
        name="lr5e8_beta1_anchor005",
        lr="5.0e-8",
        estimator_lr="5.0e-8",
        beta="1.0",
        use_anchor="true",
        anchor_weight="0.05",
        notes="Lower LR plus light anchor FM regularization.",
    ),
    Candidate(
        name="lr2e8_beta1_anchor005",
        lr="2.0e-8",
        estimator_lr="2.0e-8",
        beta="1.0",
        use_anchor="true",
        anchor_weight="0.05",
        notes="Last conservative attempt: lower LR and light anchor, beta fixed.",
    ),
]


def run_cmd(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
    )


def shell_quote_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.log_path = self.repo / args.log_path
        self.state_path = self.repo / args.state_path
        self.warmup_ckpt = self.repo / args.warmup_ckpt
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, object] = self._load_state()
        os.environ.pop("WANDB_API_KEY", None)

    def log(self, message: str) -> None:
        now = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%F %T %Z")
        line = f"[supervisor] {now} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {
                "candidate_index": 0,
                "completed_runs": [],
                "active_task": None,
                "active_run_id": None,
                "active_session": None,
                "stopped": False,
            }
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "candidate_index": 0,
                "completed_runs": [],
                "active_task": None,
                "active_run_id": None,
                "active_session": None,
                "stopped": False,
            }

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def until_reached(self) -> bool:
        until = dt.datetime.fromisoformat(self.args.until_kst)
        if until.tzinfo is None:
            until = until.replace(tzinfo=ZoneInfo("Asia/Seoul"))
        now = dt.datetime.now(ZoneInfo("Asia/Seoul"))
        return now >= until.astimezone(ZoneInfo("Asia/Seoul"))

    def wandb_api(self):
        import wandb

        return wandb.Api()

    def discover_latest_fixed_log(self) -> Path | None:
        patterns = [
            str(self.repo / "logs/test_runs/pareto_mapencfreeze_lr1e7_beta1_step200_fixedtoggle_clsft_v100x4_*_from_fakewarmup_trainb4_valb4.log"),
            str(self.repo / "logs/test_runs/auto_clsft_*_trainb*_valb*.log"),
        ]
        files: list[str] = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        if not files:
            return None
        files.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return Path(files[0])

    @staticmethod
    def extract_run_id_from_log(log_path: Path) -> str | None:
        if not log_path.exists():
            return None
        text = log_path.read_bytes()[-1_000_000:].decode("utf-8", "ignore")
        matches = re.findall(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)", text)
        if matches:
            return matches[-1]
        matches = re.findall(r"wandb: setting up run ([A-Za-z0-9_-]+)", text)
        return matches[-1] if matches else None

    @staticmethod
    def extract_task_from_log(log_path: Path) -> str | None:
        name = log_path.name
        if name.endswith(".log"):
            return name[:-4]
        return None

    def active_run_from_logs(self) -> tuple[str | None, str | None, Path | None]:
        active_task = self.state.get("active_task")
        if isinstance(active_task, str):
            p = self.repo / "logs/test_runs" / f"{active_task}.log"
            run_id = self.extract_run_id_from_log(p)
            if run_id:
                return active_task, run_id, p

        latest = self.discover_latest_fixed_log()
        if latest is None:
            return None, None, None
        run_id = self.extract_run_id_from_log(latest)
        task = self.extract_task_from_log(latest)
        if run_id and task:
            self.state["active_task"] = task
            self.state["active_run_id"] = run_id
            self.save_state()
        return task, run_id, latest

    def fetch_history(self, run_id: str) -> tuple[object, list[dict], list[dict]]:
        api = self.wandb_api()
        run = api.run(f"{self.args.wandb_entity}/{self.args.wandb_project}/{run_id}")
        val_rows = list(
            run.scan_history(
                keys=[GLOBAL_STEP_KEY, "epoch", RMM_KEY, CPD_KEY],
                page_size=1000,
            )
        )
        train_rows = list(
            run.scan_history(
                keys=[GLOBAL_STEP_KEY, ESTIMATOR_LOSS_KEY, TRAIN_LOSS_KEY],
                page_size=1000,
            )
        )
        return run, val_rows, train_rows

    def analyze(
        self,
        run: object,
        val_rows: list[dict],
        train_rows: list[dict],
        log_path: Path | None,
    ) -> tuple[str, str]:
        rmm_values = [
            float(row[RMM_KEY])
            for row in val_rows
            if row.get(RMM_KEY) is not None and math.isfinite(float(row[RMM_KEY]))
        ]
        cpd_values = [
            float(row[CPD_KEY])
            for row in val_rows
            if row.get(CPD_KEY) is not None and math.isfinite(float(row[CPD_KEY]))
        ]
        low_threshold = self.args.pretrained_rmm - self.args.rmm_low_margin
        low_count = sum(1 for value in rmm_values if value < low_threshold)
        best = max(rmm_values) if rmm_values else None
        latest = rmm_values[-1] if rmm_values else None
        latest_cpd = cpd_values[-1] if cpd_values else None

        estimator_losses = [
            float(row[ESTIMATOR_LOSS_KEY])
            for row in train_rows
            if row.get(ESTIMATOR_LOSS_KEY) is not None
        ]
        train_losses = [
            float(row[TRAIN_LOSS_KEY])
            for row in train_rows
            if row.get(TRAIN_LOSS_KEY) is not None
        ]
        if any(not math.isfinite(v) for v in estimator_losses + train_losses):
            return "stop_retry_smaller", "NaN/Inf loss detected."
        if len(estimator_losses) >= 20 and all(abs(v) < 1.0e-12 for v in estimator_losses[-20:]):
            return "stop_code_issue", "generated estimator loss stayed zero for last 20 logs."

        error_text = ""
        if log_path and log_path.exists():
            error_text = log_path.read_bytes()[-800_000:].decode("utf-8", "ignore")
        if "CUDA out of memory" in error_text or "OutOfMemoryError" in error_text:
            return "stop_retry_smaller", "OOM detected in log."
        if "Traceback (most recent call last):" in error_text or "RuntimeError:" in error_text:
            if "ProcessRaisedException" not in error_text:
                return "stop_code_issue", "Runtime error/traceback detected in log."

        self.log(
            "metrics "
            f"state={getattr(run, 'state', 'unknown')} vals={len(rmm_values)} "
            f"latest_rmm={latest} best_rmm={best} low_count={low_count} "
            f"latest_cpd={latest_cpd} train_loss_last={train_losses[-1] if train_losses else None} "
            f"estimator_loss_last={estimator_losses[-1] if estimator_losses else None}"
        )

        if low_count >= self.args.max_low_count:
            return (
                "stop_retry",
                f"RMM below {low_threshold:.6f} for {low_count} validation points.",
            )

        if len(rmm_values) >= self.args.stagnation_min_points:
            first = sum(rmm_values[:3]) / min(3, len(rmm_values))
            last = sum(rmm_values[-3:]) / min(3, len(rmm_values))
            improved = best is not None and best >= self.args.pretrained_rmm + self.args.meaningful_gain
            if not improved and last <= first + self.args.stagnation_margin:
                return (
                    "stop_retry",
                    "No meaningful RMM gain after enough validations "
                    f"(first_avg={first:.6f}, last_avg={last:.6f}, best={best:.6f}).",
                )

        if getattr(run, "state", None) in {"finished", "failed", "crashed", "killed"}:
            if best is None:
                return "stop_retry", f"Run ended as {getattr(run, 'state', None)} with no validation."
            if best < self.args.pretrained_rmm + self.args.meaningful_gain:
                return "stop_retry", f"Run ended as {getattr(run, 'state', None)} without meaningful RMM gain."
            return "keep_finished", f"Run ended with meaningful RMM gain: best={best:.6f}."

        return "keep", "Run still acceptable."

    def stop_active(self, task: str | None, reason: str) -> None:
        self.log(f"stopping active run task={task} reason={reason}")
        sessions = run_cmd(["tmux", "ls"], check=False).stdout or ""
        for line in sessions.splitlines():
            name = line.split(":", 1)[0]
            if name.startswith("pareto_clsft_fixedtoggle_v100x4_") or name.startswith("auto_clsft_"):
                run_cmd(["tmux", "kill-session", "-t", name], check=False)
        if task:
            pids = run_cmd(["pgrep", "-f", task], check=False).stdout.strip().split()
            if pids:
                run_cmd(["kill", "-TERM", *pids], check=False)
                time.sleep(30)
                pids = run_cmd(["pgrep", "-f", task], check=False).stdout.strip().split()
                if pids:
                    run_cmd(["kill", "-KILL", *pids], check=False)

    def git_refresh(self) -> bool:
        os.chdir(self.repo)
        dirty = run_cmd(["git", "status", "--porcelain"], check=False).stdout.strip()
        if dirty:
            self.log("repo dirty; refusing to auto-launch a new run.")
            return False
        pull = run_cmd(["git", "pull", "--ff-only"], check=False).stdout.strip()
        self.log(f"git pull before launch: {pull}")
        return True

    def next_candidate(self) -> Candidate | None:
        idx = int(self.state.get("candidate_index", 0))
        if idx >= len(CANDIDATES):
            return None
        self.state["candidate_index"] = idx + 1
        self.save_state()
        return CANDIDATES[idx]

    def launch_candidate(self, candidate: Candidate) -> None:
        if not self.warmup_ckpt.exists():
            self.log(f"warmup checkpoint missing, cannot launch candidate: {self.warmup_ckpt}")
            return
        if not self.git_refresh():
            return
        ts = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d_%H%M%S")
        task = f"auto_clsft_{candidate.name}_{ts}_trainb{candidate.train_b}_valb{candidate.val_b}"
        session = f"auto_clsft_{candidate.name}_{ts}"
        log_path = f"logs/test_runs/{task}.log"
        env = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "NPROC_PER_NODE": "4",
            "NUM_NODES": "1",
            "ACTION": "finetune",
            "MY_TASK_NAME": task,
            "CKPT_PATH": str(self.warmup_ckpt.relative_to(self.repo)),
            "WANDB_ENTITY": self.args.wandb_entity,
            "WANDB_PROJECT": self.args.wandb_project,
            "WANDB_MODE": "online",
            "WANDB_OFFLINE": "false",
            "MAX_EPOCHS": "16",
            "LIMIT_TRAIN_BATCHES": "1.0",
            "LIMIT_VAL_BATCHES": "0.1",
            "VAL_CHECK_INTERVAL": "200",
            "CHECK_VAL_EVERY_N_EPOCH": "null",
            "PRECISION": "32-true",
            "TRAINER_STRATEGY": "ddp_find_unused_parameters_true",
            "NUM_SANITY_VAL_STEPS": "0",
            "LOG_EVERY_N_STEPS": "1",
            "TRAIN_B": candidate.train_b,
            "VAL_B": candidate.val_b,
            "TEST_B": candidate.val_b,
            "NUM_WORKERS": "8",
            "PREFETCH_FACTOR": "2",
            "PERSISTENT_WORKERS": "true",
            "PIN_MEMORY": "true",
            "TRAIN_EPOCH_SAMPLE_FRACTION": "0.5",
            "TRAIN_USE_EVAL_AGENT_SELECTION": "true",
            "LR": candidate.lr,
            "ESTIMATOR_LR": candidate.estimator_lr,
            "LR_WARMUP_STEPS": "0",
            "LR_MIN_RATIO": "1.0",
            "DM_OBJECTIVE": "dmd",
            "DMD_BETA": candidate.beta,
            "SF_ENABLED": "true",
            "SF_START_EPOCH": "0",
            "SF_WEIGHT": "1.0",
            "SF_PATH_STEP_SIZE": "0.05",
            "USE_ANCHOR_FM": candidate.use_anchor,
            "ANCHOR_WEIGHT": candidate.anchor_weight,
            "ESTIMATOR_UPDATES_PER_STEP": "3",
            "SF_N_ROLLOUTS": "1",
            "SF_N_ANCHORS": "1",
            "SF_ANCHOR_STRIDE": "1",
            "ESTIMATOR_WARMUP_EPOCHS": "0",
            "ESTIMATOR_WARMUP_STEPS": "0",
            "SF_INIT_AUX_FROM_GEN": "false",
            "SF_UNFROZEN_RANGE": "except_map_encoder",
            "SF_EMA_WEIGHT": "0.0",
            "SF_EMA_START_STEP": "1000000000",
            "SF_GRAD_CLIP": "1.0",
            "SAMPLING_SAMPLE_STEPS": "16",
            "SAMPLING_SAMPLE_METHOD": "euler",
            "SAMPLING_NOISE_SCALE": "1.0",
            "SAMPLING_RTS_ENABLED": "true",
            "SAMPLING_RTS_POLICY": "all",
            "SAMPLING_RTS_MIN_EXECUTED_STEPS": "16",
            "SAMPLING_RTS_BACKPROP_LAST_K": "8",
            "SAMPLING_RTS_SCOPE": "global_batch",
            "VAL_SAMPLE_STEPS": "16",
            "VAL_SAMPLE_METHOD": "euler",
            "VAL_NOISE_SCALE": "1.0",
            "N_ROLLOUT_CLOSED_VAL": "16",
            "N_BATCH_SIM_AGENTS_METRIC": "100000",
            "SCORER_SCENE_NUM": "1680",
            "SIM_AGENTS_METRIC_WORKERS": "0",
            "VAL_OPEN_LOOP": "true",
            "VAL_CLOSED_LOOP": "true",
            "N_VIS_BATCH": "0",
            "N_VIS_SCENARIO": "0",
            "N_VIS_ROLLOUT": "0",
            "DELETE_LOCAL_VIDEOS_AFTER_UPLOAD": "true",
            "CLOSED_LOOP_ROLLOUT_MODE": "raw_fm",
            "DECODER_USE_LQR": "false",
            "WOSAC_CPD_REFERENCE": "null",
            "CHECKPOINT_MONITOR": RMM_KEY,
            "CHECKPOINT_MODE": "max",
            "CHECKPOINT_SAVE_TOP_K": "1",
            "WANDB_LOG_MODEL": "all",
            "EXTRA_ARGS": (
                "+model.model_config.self_forced.allow_auxiliary_finetune=true "
                "+callbacks.self_forced_warmup_validation_gate._target_=src.utils.self_forced_warmup_validation_gate.SelfForcedWarmupValidationGateCallback "
                "+callbacks.self_forced_warmup_validation_gate.verbose=false"
            ),
        }
        env.update(candidate.extra)
        command = (
            f"cd {shlex.quote(str(self.repo))} && "
            f"{shell_quote_env(env)} "
            f"env -u WANDB_API_KEY bash scripts/train_self_forced_npfm_pareto.sh "
            f"2>&1 | tee {shlex.quote(log_path)}"
        )
        run_cmd(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", command], check=True)
        self.state["active_task"] = task
        self.state["active_run_id"] = None
        self.state["active_session"] = session
        self.save_state()
        self.log(f"launched candidate={candidate.name} task={task} session={session} notes={candidate.notes}")

    def halve_batch_retry(self) -> None:
        candidate = Candidate(
            name="oom_retry_b2",
            lr="5.0e-8",
            estimator_lr="5.0e-8",
            beta="1.0",
            train_b="2",
            val_b="2",
            notes="OOM/NaN retry with half batch.",
        )
        self.launch_candidate(candidate)

    def loop_once(self) -> None:
        if not self.warmup_ckpt.exists():
            self.log("waiting for warmup checkpoint before supervising fixed runs")
            return

        task, run_id, log_path = self.active_run_from_logs()
        if not run_id:
            self.log("waiting for fixed/auto run W&B id")
            return

        try:
            run, val_rows, train_rows = self.fetch_history(run_id)
        except Exception as exc:
            self.log(f"wandb fetch failed for {run_id}: {type(exc).__name__}: {exc}")
            return

        decision, reason = self.analyze(run, val_rows, train_rows, log_path)
        self.log(f"decision={decision} run_id={run_id} task={task} reason={reason}")
        if decision == "keep":
            return
        if decision == "keep_finished":
            self.state["stopped"] = True
            self.save_state()
            return

        self.stop_active(task, reason)
        completed = list(self.state.get("completed_runs", []))
        completed.append({"run_id": run_id, "task": task, "decision": decision, "reason": reason})
        self.state["completed_runs"] = completed
        self.state["active_task"] = None
        self.state["active_run_id"] = None
        self.state["active_session"] = None
        self.save_state()

        if decision == "stop_code_issue":
            self.log("code issue suspected; not relaunching automatically.")
            self.state["stopped"] = True
            self.save_state()
            return
        if decision == "stop_retry_smaller":
            self.halve_batch_retry()
            return

        candidate = self.next_candidate()
        if candidate is None:
            self.log("candidate queue exhausted; stopping supervisor.")
            self.state["stopped"] = True
            self.save_state()
            return
        self.launch_candidate(candidate)

    def run(self) -> None:
        self.log(
            "started "
            f"pretrained_rmm={self.args.pretrained_rmm} "
            f"low_threshold={self.args.pretrained_rmm - self.args.rmm_low_margin} "
            f"max_low_count={self.args.max_low_count} until={self.args.until_kst}"
        )
        while not self.until_reached():
            if self.state.get("stopped"):
                self.log("state stopped=true; exiting")
                return
            try:
                self.loop_once()
            except Exception as exc:
                self.log(f"supervisor loop error: {type(exc).__name__}: {exc}")
            time.sleep(self.args.interval_seconds)
        self.log("until_kst reached; exiting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/mnt/nuplan/projects/catk")
    parser.add_argument("--wandb-entity", default="se99an")
    parser.add_argument("--wandb-project", default="clsft-catk")
    parser.add_argument("--pretrained-rmm", type=float, default=0.7792712450027466)
    parser.add_argument("--pretrained-cpd", type=float, default=0.20425976812839508)
    parser.add_argument("--rmm-low-margin", type=float, default=0.001)
    parser.add_argument("--max-low-count", type=int, default=3)
    parser.add_argument("--meaningful-gain", type=float, default=0.002)
    parser.add_argument("--stagnation-min-points", type=int, default=5)
    parser.add_argument("--stagnation-margin", type=float, default=0.0005)
    parser.add_argument("--interval-seconds", type=int, default=180)
    parser.add_argument("--until-kst", default="2026-06-01T09:00:00+09:00")
    parser.add_argument(
        "--warmup-ckpt",
        default=(
            "logs/pareto_mapencfreeze_lr1e7_beta1_step200_skipfake_clsft_pareto_clsft_v100x4_0529_162458_fake1ep_trainb4_valb4/"
            "runs/2026-05-29_16-52-48/checkpoints/fake_warmup_epoch0.ckpt"
        ),
    )
    parser.add_argument(
        "--log-path",
        default="logs/test_runs/clsft_auto_supervisor_0529_162458.log",
    )
    parser.add_argument(
        "--state-path",
        default="logs/test_runs/clsft_auto_supervisor_0529_162458.state.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        Supervisor(parse_args()).run()
    except KeyboardInterrupt:
        sys.exit(130)
