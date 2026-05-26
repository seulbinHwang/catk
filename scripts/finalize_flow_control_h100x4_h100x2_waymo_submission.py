#!/usr/bin/env python3
"""Recover, archive, and optionally upload an H100x4+H100x2 Flow Waymo submission.

The validation rollout writes rank-local submission shards on each pod. This
helper does not rerun rollout and does not create/delete pods. It copies the
worker pod shards into the rank-0 pod collection directory, verifies the shard
set, builds the Waymo tar.gz archive, and can invoke the existing Waymo uploader.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def _kubectl_exec(args: argparse.Namespace, pod: str, shell: str, *, check: bool = True) -> str:
    return _run(
        [
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
            shell,
        ],
        check=check,
    ).stdout


def _pod_ip(args: argparse.Namespace, pod: str) -> str:
    return _run(
        [
            "kubectl",
            "get",
            "pod",
            "-n",
            args.namespace,
            pod,
            "-o",
            "jsonpath={.status.podIP}",
        ]
    ).stdout.strip()


def _copy_shards(args: argparse.Namespace) -> None:
    master_ip = _pod_ip(args, args.master_pod)
    run_dir = args.run_dir.rstrip("/")
    source_dir = f"{run_dir}/{args.submission_dirname}"
    collect_dir = f"{run_dir}/{args.collect_dirname}"

    prepare_master = textwrap.dedent(
        f"""
        set -Eeuo pipefail
        rm -rf {collect_dir!r}
        mkdir -p {collect_dir!r}
        cp -f {source_dir!r}/submission-rank0[0-3]-*.binproto {collect_dir!r}/
        """
    ).strip()
    print(_kubectl_exec(args, args.master_pod, prepare_master), end="")

    receiver_shell = textwrap.dedent(
        f"""
        set -Eeuo pipefail
        cd {collect_dir!r}
        timeout {args.copy_timeout_seconds:d} nc -l -p {args.port:d} | tar -xpf -
        """
    ).strip()
    receiver = subprocess.Popen(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            args.master_pod,
            "-c",
            args.container,
            "--",
            "bash",
            "-lc",
            receiver_shell,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    sender_shell = textwrap.dedent(
        f"""
        set -Eeuo pipefail
        cd {source_dir!r}
        shopt -s nullglob
        files=({args.worker_shard_glob})
        if [[ ${{#files[@]}} -eq 0 ]]; then
          echo "no worker shards matched: {args.worker_shard_glob}" >&2
          exit 2
        fi
        if [[ {args.expected_remote_shards:d} -gt 0 && ${{#files[@]}} -ne {args.expected_remote_shards:d} ]]; then
          echo "expected {args.expected_remote_shards:d} remote shards, got ${{#files[@]}}" >&2
          exit 2
        fi
        tar -cf - "${{files[@]}}" | nc -N {master_ip} {args.port:d}
        """
    ).strip()
    sender_out = _kubectl_exec(args, args.worker_pod, sender_shell, check=False)
    receiver_out, _ = receiver.communicate(timeout=args.copy_timeout_seconds + 60)
    if sender_out:
        print(sender_out, end="")
    if receiver_out:
        print(receiver_out, end="")
    if receiver.returncode != 0:
        raise SystemExit(f"receiver exited with status {receiver.returncode}")


def _verify_and_archive(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.rstrip("/")
    collect_dir = f"{run_dir}/{args.collect_dirname}"
    archive_path = args.archive_path or f"{run_dir}/sim_agents_2025_submission.tar.gz"
    verify_members = str(args.verify_archive_members).lower()
    shell = textwrap.dedent(
        f"""
        set -Eeuo pipefail
        python - <<'PY'
        import os
        import shutil
        import subprocess
        import tarfile
        from collections import Counter
        from pathlib import Path

        collect = Path({collect_dir!r})
        archive = Path({archive_path!r})
        shards = sorted(collect.glob("*.binproto"))
        if {args.expected_total_shards:d} > 0 and len(shards) != {args.expected_total_shards:d}:
            raise SystemExit(f"expected {args.expected_total_shards:d} shards, got {{len(shards)}}")
        if not shards:
            raise SystemExit("no submission shards found")
        small = [p for p in shards if p.stat().st_size <= {args.min_shard_bytes:d}]
        if small:
            raise SystemExit("suspiciously small shard files: " + ", ".join(p.name for p in small[:8]))
        rank_counts = Counter(p.name.split("-")[1] for p in shards)
        missing_ranks = [f"rank{{rank:02d}}" for rank in range(6) if rank_counts.get(f"rank{{rank:02d}}", 0) == 0]
        if missing_ranks:
            raise SystemExit("missing rank shards: " + ", ".join(missing_ranks))

        archive.unlink(missing_ok=True)
        total = len(shards)
        pigz = shutil.which("pigz")
        if pigz:
            links = archive.parent / f".{{archive.name}}.links.tmp"
            if links.exists():
                shutil.rmtree(links)
            links.mkdir(parents=True, exist_ok=True)
            try:
                names = []
                for i, shard in enumerate(shards):
                    name = f"submission.binproto-{{i:05d}}-of-{{total:05d}}"
                    os.link(shard, links / name)
                    names.append(name)
                with archive.open("wb") as out:
                    tar_proc = subprocess.Popen(["tar", "-C", links.as_posix(), "-cf", "-", *names], stdout=subprocess.PIPE)
                    assert tar_proc.stdout is not None
                    pigz_proc = subprocess.Popen([pigz, "-{args.compresslevel:d}"], stdin=tar_proc.stdout, stdout=out)
                    tar_proc.stdout.close()
                    pigz_rc = pigz_proc.wait()
                    tar_rc = tar_proc.wait()
                if tar_rc != 0 or pigz_rc != 0:
                    raise SystemExit(f"tar/pigz failed: tar_rc={{tar_rc}} pigz_rc={{pigz_rc}}")
            finally:
                shutil.rmtree(links, ignore_errors=True)
        else:
            with tarfile.open(archive, "w:gz", compresslevel={args.compresslevel:d}) as tar:
                for i, shard in enumerate(shards):
                    tar.add(shard.as_posix(), arcname=f"submission.binproto-{{i:05d}}-of-{{total:05d}}")
        print(f"archive={{archive}} bytes={{archive.stat().st_size}} shards={{len(shards)}} ranks={{dict(sorted(rank_counts.items()))}}")
        PY
        if [[ {verify_members!r} == "true" ]]; then
          tar -tzf {archive_path!r} >/tmp/catk_waymo_flow_archive_members.txt
          if [[ {args.expected_total_shards:d} -gt 0 ]]; then
            test "$(wc -l </tmp/catk_waymo_flow_archive_members.txt)" -eq {args.expected_total_shards:d}
          fi
          head -n 1 /tmp/catk_waymo_flow_archive_members.txt
          tail -n 1 /tmp/catk_waymo_flow_archive_members.txt
        fi
        """
    ).strip()
    print(_kubectl_exec(args, args.master_pod, shell), end="")


def _verify_existing_archive(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.rstrip("/")
    archive_path = args.archive_path or f"{run_dir}/sim_agents_2025_submission.tar.gz"
    verify_members = str(args.verify_archive_members).lower()
    shell = textwrap.dedent(
        f"""
        set -Eeuo pipefail
        test -f {archive_path!r}
        ls -lh {archive_path!r}
        if [[ {verify_members!r} == "true" ]]; then
          tar -tzf {archive_path!r} >/tmp/catk_waymo_flow_archive_members.txt
          head -n 1 /tmp/catk_waymo_flow_archive_members.txt
          tail -n 1 /tmp/catk_waymo_flow_archive_members.txt
        fi
        """
    ).strip()
    print(_kubectl_exec(args, args.master_pod, shell), end="")


def _upload(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.rstrip("/")
    archive_path = args.archive_path or f"{run_dir}/sim_agents_2025_submission.tar.gz"
    storage_state = args.storage_state_path or f"{args.project_root.rstrip('/')}/secrets/waymo/waymo_storage_state.json"
    upload_code = f"""
from omegaconf import OmegaConf
from src.utils.waymo_submission import maybe_submit_waymo_submission

cfg = OmegaConf.create({{
    "action": "validate",
    "paths": {{"root_dir": {args.project_root!r}, "output_dir": {run_dir!r}}},
    "model": {{"model_config": {{"sim_agents_submission": {{"method_name": {args.method_name!r}}}}}}},
    "waymo_submission": {{
        "enabled": True,
        "submit_validate": True,
        "submit_test": False,
        "evaluation_set": None,
        "challenge_url": "https://waymo.com/open/challenges/2025/sim-agents/",
        "submissions_url": "https://waymo.com/open/challenges/submissions/",
        "storage_state_path": {storage_state!r},
        "archive_path": {archive_path!r},
        "browser_name": "chromium",
        "browser_channel": None,
        "browser_executable_path": None,
        "headless": True,
        "chromium_sandbox": False,
        "navigation_timeout_ms": 120000,
        "upload_timeout_ms": {args.upload_timeout_ms:d},
        "post_submit_wait_ms": 5000,
        "poll_submission_status": False,
        "poll_timeout_seconds": 0,
        "poll_interval_seconds": 60,
        "save_debug_artifacts": True,
    }},
}})
print(maybe_submit_waymo_submission(cfg))
"""
    shell = (
        "set -Eeuo pipefail\n"
        f"cd {args.project_root!r}\n"
        f"test -f {storage_state!r}\n"
        f"test -f {archive_path!r}\n"
        f"PYTHONPATH={args.project_root!r} python - <<'PY'\n"
        f"{upload_code.strip()}\n"
        "PY\n"
    )
    print(_kubectl_exec(args, args.master_pod, shell), end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Full run directory on both pods.")
    parser.add_argument("--project-root", default="/mnt/nuplan/projects/catk")
    parser.add_argument("--namespace", default="p-pnc")
    parser.add_argument("--container", default="main")
    parser.add_argument("--master-pod", default="hsb-npc-training")
    parser.add_argument("--worker-pod", default="wo-pvc-2")
    parser.add_argument("--submission-dirname", default="sim_agents_2025_submission")
    parser.add_argument("--collect-dirname", default="sim_agents_2025_submission_rank0_collect")
    parser.add_argument("--worker-shard-glob", default="submission-rank0[4-5]-*.binproto")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--storage-state-path", default=None)
    parser.add_argument("--method-name", default="SMART-flow-control epoch061 H100x6")
    parser.add_argument("--port", type=int, default=29654)
    parser.add_argument("--copy-timeout-seconds", type=int, default=900)
    parser.add_argument("--expected-remote-shards", type=int, default=0)
    parser.add_argument("--expected-total-shards", type=int, default=0)
    parser.add_argument("--min-shard-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--upload-timeout-ms", type=int, default=7200000)
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--skip-archive", action="store_true")
    parser.add_argument("--verify-archive-members", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not args.skip_copy:
        _copy_shards(args)
    if args.skip_archive:
        _verify_existing_archive(args)
    else:
        _verify_and_archive(args)
    if args.upload:
        _upload(args)


if __name__ == "__main__":
    main()
