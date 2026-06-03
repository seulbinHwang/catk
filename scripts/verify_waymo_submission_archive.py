#!/usr/bin/env python3
"""Verify a Waymo Sim Agents 2025 submission archive.

This checker is intentionally independent from the rollout job. It opens the
final ``sim_agents_2025_submission.tar.gz``, parses every ``binproto`` member,
and checks the invariants that matter before spending a leaderboard submission:

* the tar member naming is contiguous and self-consistent;
* every shard parses as ``SimAgentsChallengeSubmission``;
* submission metadata matches the intended method/account fields;
* every scenario id is unique across the archive;
* every scenario has the expected number of joint rollouts;
* every simulated trajectory has the expected future length and finite values.
"""

from __future__ import annotations

import argparse
import math
import re
import tarfile
from collections import Counter
from pathlib import Path


_MEMBER_RE = re.compile(r"^submission\.binproto-(\d{5})-of-(\d{5})$")


def _parse_authors(value: str | None) -> list[str] | None:
    if value in (None, ""):
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _assert_equal(name: str, actual: object, expected: object | None) -> None:
    if expected is None:
        return
    if actual != expected:
        raise SystemExit(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _check_finite(values, *, label: str) -> None:
    for value in values:
        if not math.isfinite(float(value)):
            raise SystemExit(f"non-finite value in {label}: {value!r}")


def _verify_member_name(member_name: str, expected_total: int | None) -> tuple[int, int]:
    match = _MEMBER_RE.match(member_name)
    if match is None:
        raise SystemExit(f"invalid archive member name: {member_name!r}")
    index = int(match.group(1))
    total = int(match.group(2))
    if expected_total is not None and total != expected_total:
        raise SystemExit(
            f"member {member_name!r} declares total={total}, expected {expected_total}"
        )
    return index, total


def _load_sim_agents_submission_pb2():
    try:
        from waymo_open_dataset.protos import sim_agents_submission_pb2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "waymo_open_dataset is required to parse Sim Agents submission protos. "
            "Run this verifier inside the CATK training environment."
        ) from exc
    return sim_agents_submission_pb2


def verify_archive(args: argparse.Namespace) -> None:
    archive_path = Path(args.archive).expanduser().resolve()
    if not archive_path.is_file():
        raise SystemExit(f"archive not found: {archive_path}")

    sim_agents_submission_pb2 = _load_sim_agents_submission_pb2()
    expected_authors = _parse_authors(args.expected_authors)
    scenario_ids: set[str] = set()
    scenario_count = 0
    member_indices: list[int] = []
    declared_totals: set[int] = set()
    shard_scenario_counts: Counter[int] = Counter()

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        if not members:
            raise SystemExit("archive contains no file members")
        if args.expected_shards is not None and len(members) != args.expected_shards:
            raise SystemExit(
                f"expected {args.expected_shards} archive members, got {len(members)}"
            )

        for member in members:
            index, declared_total = _verify_member_name(
                member.name,
                args.expected_shards,
            )
            member_indices.append(index)
            declared_totals.add(declared_total)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise SystemExit(f"failed to extract archive member: {member.name}")

            submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission()
            submission.ParseFromString(extracted.read())

            expected_type = (
                sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION
            )
            if submission.submission_type != expected_type:
                raise SystemExit(
                    f"{member.name}: submission_type={submission.submission_type}, "
                    f"expected SIM_AGENTS_SUBMISSION"
                )

            _assert_equal("unique_method_name", submission.unique_method_name, args.expected_method_name)
            if expected_authors is not None:
                _assert_equal("authors", list(submission.authors), expected_authors)
            _assert_equal("affiliation", submission.affiliation, args.expected_affiliation)
            _assert_equal("description", submission.description, args.expected_description)
            _assert_equal("method_link", submission.method_link, args.expected_method_link)
            _assert_equal("account_name", submission.account_name, args.expected_account_name)
            _assert_equal("num_model_parameters", submission.num_model_parameters, args.expected_num_model_parameters)
            if args.require_closed_loop_ack and not bool(
                submission.acknowledge_complies_with_closed_loop_requirement
            ):
                raise SystemExit(f"{member.name}: closed-loop acknowledgement is false")

            shard_scenario_counts[index] = len(submission.scenario_rollouts)
            for scenario in submission.scenario_rollouts:
                if not scenario.scenario_id:
                    raise SystemExit(f"{member.name}: empty scenario_id")
                if scenario.scenario_id in scenario_ids:
                    raise SystemExit(f"duplicate scenario_id: {scenario.scenario_id}")
                scenario_ids.add(scenario.scenario_id)
                scenario_count += 1

                n_joint_scenes = len(scenario.joint_scenes)
                if n_joint_scenes != args.expected_rollouts_per_scenario:
                    raise SystemExit(
                        f"{scenario.scenario_id}: expected "
                        f"{args.expected_rollouts_per_scenario} rollouts, got {n_joint_scenes}"
                    )
                for rollout_index, joint_scene in enumerate(scenario.joint_scenes):
                    if not joint_scene.simulated_trajectories:
                        raise SystemExit(
                            f"{scenario.scenario_id}: rollout {rollout_index} has no trajectories"
                        )
                    agent_ids = set()
                    for trajectory in joint_scene.simulated_trajectories:
                        if trajectory.object_id in agent_ids:
                            raise SystemExit(
                                f"{scenario.scenario_id}: duplicate object_id "
                                f"{trajectory.object_id} in rollout {rollout_index}"
                            )
                        agent_ids.add(trajectory.object_id)
                        lengths = {
                            "center_x": len(trajectory.center_x),
                            "center_y": len(trajectory.center_y),
                            "center_z": len(trajectory.center_z),
                            "heading": len(trajectory.heading),
                        }
                        for field_name, length in lengths.items():
                            if length != args.expected_steps_per_trajectory:
                                raise SystemExit(
                                    f"{scenario.scenario_id}: rollout {rollout_index} "
                                    f"object {trajectory.object_id} field {field_name} "
                                    f"has length {length}, expected "
                                    f"{args.expected_steps_per_trajectory}"
                                )
                        _check_finite(
                            trajectory.center_x,
                            label=f"{scenario.scenario_id}/center_x",
                        )
                        _check_finite(
                            trajectory.center_y,
                            label=f"{scenario.scenario_id}/center_y",
                        )
                        _check_finite(
                            trajectory.center_z,
                            label=f"{scenario.scenario_id}/center_z",
                        )
                        _check_finite(
                            trajectory.heading,
                            label=f"{scenario.scenario_id}/heading",
                        )

    if declared_totals != {len(member_indices)}:
        raise SystemExit(
            f"archive members declare totals {sorted(declared_totals)}, "
            f"but member count is {len(member_indices)}"
        )
    expected_indices = list(range(len(member_indices)))
    if sorted(member_indices) != expected_indices:
        raise SystemExit(
            f"archive member indices are not contiguous: got {sorted(member_indices)}, "
            f"expected {expected_indices}"
        )
    if args.expected_scenarios is not None and scenario_count != args.expected_scenarios:
        raise SystemExit(
            f"expected {args.expected_scenarios} scenarios, got {scenario_count}"
        )
    if scenario_count < args.min_scenarios:
        raise SystemExit(
            f"expected at least {args.min_scenarios} scenarios, got {scenario_count}"
        )

    print(
        "OK "
        f"archive={archive_path} "
        f"members={len(member_indices)} "
        f"scenarios={scenario_count} "
        f"rollouts_per_scenario={args.expected_rollouts_per_scenario} "
        f"steps_per_trajectory={args.expected_steps_per_trajectory} "
        f"shard_scenario_counts={dict(sorted(shard_scenario_counts.items()))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--expected-shards", type=int, default=None)
    parser.add_argument("--expected-scenarios", type=int, default=None)
    parser.add_argument("--min-scenarios", type=int, default=1)
    parser.add_argument("--expected-rollouts-per-scenario", type=int, default=32)
    parser.add_argument("--expected-steps-per-trajectory", type=int, default=80)
    parser.add_argument("--expected-method-name", default=None)
    parser.add_argument("--expected-authors", default=None, help="Comma-separated author list.")
    parser.add_argument("--expected-affiliation", default=None)
    parser.add_argument("--expected-description", default=None)
    parser.add_argument("--expected-method-link", default=None)
    parser.add_argument("--expected-account-name", default=None)
    parser.add_argument("--expected-num-model-parameters", default=None)
    parser.add_argument("--require-closed-loop-ack", action="store_true")
    return parser.parse_args()


def main() -> None:
    verify_archive(parse_args())


if __name__ == "__main__":
    main()
