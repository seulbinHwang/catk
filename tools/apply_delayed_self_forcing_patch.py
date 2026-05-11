from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


README_MARKER = "### Delayed-Window Self-Forcing"
README_SECTION = """

### Delayed-Window Self-Forcing

이 모드는 self-forcing fine-tuning에서 학습 시작 시점만 epoch에 따라 뒤로 미룹니다.
RMM을 직접 loss로 쓰지 않고, 기존 self-forcing loss를 그대로 사용합니다.

| epoch | 전체 rollout | 학습 제외 구간 | 실제 학습 구간 | 기준 시점 |
|---:|---:|---:|---:|---:|
| 0~3 | 2초 | 없음 | 0~2초 | 0초 |
| 4~7 | 4초 | 0~2초 | 2~4초 | 2초 |
| 8~11 | 6초 | 0~4초 | 4~6초 | 4초 |
| 12~15 | 8초 | 0~6초 | 6~8초 | 6초 |

핵심 규칙은 아래와 같습니다.

- 앞구간은 현재 모델이 스스로 굴러가게만 하고 loss에는 쓰지 않습니다.
- 앞구간과 실제 학습 구간 사이의 gradient 연결은 끊습니다.
- 실제 학습 2초 구간 안의 0.5초 block 연결은 유지합니다.
- target horizon은 항상 2초입니다. 그래서 `decoder.flow_window_steps=20`을 사용합니다.
- 새 RMM loss, 혼합 window, random window는 추가하지 않습니다.

6x H100 실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
torchrun --standalone --nproc_per_node=6 -m src.run \\
  experiment=self_forced_delayed_npfm_h100_6 \\
  action=finetune \\
  paths.cache_root="$CACHE_ROOT" \\
  task_name=self_forced_delayed_window_h100_6 \\
  ckpt_path="/path/to/pretrained.ckpt"
```

4x H100 실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
torchrun --standalone --nproc_per_node=4 -m src.run \\
  experiment=self_forced_delayed_npfm_h100_4 \\
  action=finetune \\
  paths.cache_root="$CACHE_ROOT" \\
  task_name=self_forced_delayed_window_h100_4 \\
  ckpt_path="/path/to/pretrained.ckpt"
```

상세 설명은 `docs/DELAYED_SELF_FORCING_USAGE.md`에 있습니다.
"""


def find_repo_root(start: Path) -> Path:
    """현재 위치에서 CAT-K 저장소 루트를 찾습니다.

    Args:
        start: 검색을 시작할 경로입니다.

    Returns:
        Path: `src/smart/model/smart_flow.py`가 있는 저장소 루트입니다.

    Raises:
        FileNotFoundError: 현재 위치와 상위 경로에서 저장소 루트를 찾지 못한 경우입니다.
    """
    current = start.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "src/smart/model/smart_flow.py").exists():
            return candidate
    raise FileNotFoundError(
        "CAT-K repo root was not found. Run this script from the self_forcing_2 branch root."
    )


def read_text(path: Path) -> str:
    """UTF-8 텍스트 파일을 읽습니다.

    Args:
        path: 읽을 파일 경로입니다.

    Returns:
        str: 파일 내용입니다.
    """
    return path.read_text(encoding="utf-8")


def write_text_if_changed(path: Path, text: str) -> None:
    """내용이 달라졌을 때만 파일을 저장합니다.

    Args:
        path: 저장할 파일 경로입니다.
        text: 저장할 새 내용입니다.

    Returns:
        None
    """
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    if old_text != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """문자열을 한 번만 교체합니다.

    Args:
        text: 전체 파일 내용입니다.
        old: 찾을 문자열입니다.
        new: 바꿀 문자열입니다.
        label: 실패했을 때 표시할 설명입니다.

    Returns:
        str: 교체된 파일 내용입니다.

    Raises:
        RuntimeError: 대상 문자열이 없거나 여러 번 등장하는 경우입니다.
    """
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"Patch target not found: {label}")
    if count > 1:
        raise RuntimeError(f"Patch target is ambiguous ({count} matches): {label}")
    return text.replace(old, new, 1)


def replace_exact_count(text: str, old: str, new: str, expected_count: int, label: str) -> str:
    """문자열을 기대한 횟수만큼 교체합니다.

    Args:
        text: 전체 파일 내용입니다.
        old: 찾을 문자열입니다.
        new: 바꿀 문자열입니다.
        expected_count: 정확히 기대하는 등장 횟수입니다.
        label: 실패했을 때 표시할 설명입니다.

    Returns:
        str: 교체된 파일 내용입니다.

    Raises:
        RuntimeError: 등장 횟수가 기대와 다른 경우입니다.
    """
    count = text.count(old)
    if count != expected_count:
        raise RuntimeError(
            f"Patch target count mismatch for {label}: expected {expected_count}, got {count}."
        )
    return text.replace(old, new, expected_count)


def insert_once_before(text: str, marker: str, insert: str, label: str) -> str:
    """기준 문자열 앞에 내용을 한 번 삽입합니다.

    Args:
        text: 전체 파일 내용입니다.
        marker: 삽입 위치를 찾을 기준 문자열입니다.
        insert: 삽입할 내용입니다.
        label: 실패했을 때 표시할 설명입니다.

    Returns:
        str: 삽입된 파일 내용입니다.
    """
    if insert in text:
        return text
    return replace_once(text=text, old=marker, new=insert + marker, label=label)


def find_call_span(text: str, call_start: str) -> tuple[int, int]:
    """함수 호출의 시작과 끝 위치를 찾습니다.

    Args:
        text: 전체 파일 내용입니다.
        call_start: 찾을 호출 시작 문자열입니다. 예: `self.encoder.training_rollout_from_cache(`.

    Returns:
        tuple[int, int]: 호출 시작 index와 닫는 괄호 다음 index입니다.

    Raises:
        RuntimeError: 호출을 찾지 못하거나 괄호가 닫히지 않은 경우입니다.
    """
    start = text.find(call_start)
    if start < 0:
        raise RuntimeError(f"Call not found: {call_start}")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise RuntimeError(f"Call was not closed: {call_start}")


def patch_smart_flow_py(repo_root: Path) -> None:
    """SMARTFlow에 delayed-window self-forcing 연결을 추가합니다.

    Args:
        repo_root: CAT-K 저장소 루트입니다.

    Returns:
        None
    """
    path = repo_root / "src/smart/model/smart_flow.py"
    text = read_text(path)

    delayed_import = (
        "from src.smart.modules.self_forced_delayed_window import (\n"
        "    SelfForcedDelayedWindow,\n"
        "    build_delayed_anchor0_tokenized_agent,\n"
        "    build_delayed_normalized_committed_path,\n"
        "    resolve_self_forced_delayed_window,\n"
        ")\n"
    )
    if "self_forced_delayed_window" not in text:
        marker = "from src.smart.modules.self_forced_dmd_guidance import build_clean_dmd_direction\n"
        text = insert_once_before(
            text=text,
            marker=marker,
            insert=delayed_import,
            label="smart_flow delayed import",
        )
    text = text.replace("    build_anchor0_normalized_committed_path,\n", "")

    attr_insert = (
        "        self.self_forced_delayed_window_config = (\n"
        "            getattr(self.self_forced_config, \"delayed_window\", None)\n"
        "            if self.self_forced_config is not None\n"
        "            else None\n"
        "        )\n"
        "        self.self_forced_delayed_window_enabled = bool(\n"
        "            self.self_forced_delayed_window_config is not None\n"
        "            and getattr(self.self_forced_delayed_window_config, \"enabled\", False)\n"
        "        )\n"
        "        self.self_forced_delayed_window_stage_epochs = (\n"
        "            max(1, int(getattr(self.self_forced_delayed_window_config, \"stage_epochs\", 4)))\n"
        "            if self.self_forced_delayed_window_config is not None\n"
        "            else 4\n"
        "        )\n"
    )
    if "self_forced_delayed_window_enabled" not in text:
        marker = "        self.self_forced_weight = (\n"
        text = insert_once_before(
            text=text,
            marker=marker,
            insert=attr_insert,
            label="smart_flow delayed config attrs",
        )

    method_insert = (
        "    def _resolve_self_forced_delayed_window(self) -> SelfForcedDelayedWindow:\n"
        "        \"\"\"현재 epoch에서 self-forcing이 학습할 2초 구간을 정합니다.\n\n"
        "        Returns:\n"
        "            SelfForcedDelayedWindow: 전체 rollout 길이와 실제 학습 시작 시점을 담은 값입니다.\n"
        "        \"\"\"\n"
        "        return resolve_self_forced_delayed_window(\n"
        "            current_epoch=int(self.current_epoch),\n"
        "            start_epoch=int(self.self_forced_start_epoch),\n"
        "            flow_window_steps=int(self.flow_window_steps),\n"
        "            commit_steps=int(self.encoder.agent_encoder.shift),\n"
        "            stage_epochs=int(self.self_forced_delayed_window_stage_epochs),\n"
        "            enabled=bool(self.self_forced_delayed_window_enabled),\n"
        "        )\n\n"
    )
    if "def _resolve_self_forced_delayed_window" not in text:
        marker = "    def _should_enable_fit_time_checkpoint_only_validation(self) -> bool:\n"
        text = insert_once_before(
            text=text,
            marker=marker,
            insert=method_insert,
            label="smart_flow delayed window resolver method",
        )

    old_anchor = "        anchor_mask = get_anchor0_valid_mask(tokenized_agent)\n"
    new_anchor = (
        "        delayed_window = self._resolve_self_forced_delayed_window()\n"
        "        if self.self_forced_delayed_window_enabled:\n"
        "            anchor_mask = tokenized_agent[\"flow_eval_mask\"][:, delayed_window.anchor_offset].bool()\n"
        "        else:\n"
        "            anchor_mask = get_anchor0_valid_mask(tokenized_agent)\n"
    )
    if "tokenized_agent[\"flow_eval_mask\"][:, delayed_window.anchor_offset]" not in text:
        text = replace_once(
            text=text,
            old=old_anchor,
            new=new_anchor,
            label="smart_flow anchor mask delayed selection",
        )

    rollout_window_line = "        delayed_window = self._resolve_self_forced_delayed_window()\n"
    rollout_window_marker = "        encoder_modes = self._switch_module_to_eval_preserving_modes(self.encoder)\n"
    if (
        "rollout_steps_2hz=delayed_window.rollout_steps_2hz" not in text
        and rollout_window_line + rollout_window_marker not in text
    ):
        text = replace_once(
            text=text,
            old=rollout_window_marker,
            new=rollout_window_line + rollout_window_marker,
            label="smart_flow delayed rollout window resolver",
        )
    elif rollout_window_line + rollout_window_marker not in text:
        text = replace_once(
            text=text,
            old=rollout_window_marker,
            new=rollout_window_line + rollout_window_marker,
            label="smart_flow delayed rollout window resolver",
        )

    call_start = "self.encoder.training_rollout_from_cache("
    call_begin, call_end = find_call_span(text, call_start)
    call_text = text[call_begin:call_end]
    if "learning_start_step_2hz" not in call_text:
        call_text = re.sub(
            r"rollout_steps_2hz\s*=\s*[^,\n]+,",
            "rollout_steps_2hz=delayed_window.rollout_steps_2hz,",
            call_text,
            count=1,
        )
        if "rollout_steps_2hz=delayed_window.rollout_steps_2hz" not in call_text:
            call_text = call_text.replace(
                "self_forced_epoch=",
                "rollout_steps_2hz=delayed_window.rollout_steps_2hz,\n                self_forced_epoch=",
                1,
            )
        call_text = call_text.replace(
            "self_forced_epoch=",
            "learning_start_step_2hz=delayed_window.skipped_blocks_2hz,\n                self_forced_epoch=",
            1,
        )
        text = text[:call_begin] + call_text + text[call_end:]

    if "build_delayed_anchor0_tokenized_agent(" not in text:
        pattern = re.compile(
            r"(?P<indent>^[ \t]*)(?P<target_name>\w+)\s*=\s*build_anchor0_normalized_committed_path\(\s*\n(?P<body>.*?^[ \t]*\)\s*)",
            flags=re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(text)
        if match is None:
            raise RuntimeError("Could not find build_anchor0_normalized_committed_path call in smart_flow.py")
        body = match.group("body")
        rollout_match = re.search(r"pred_traj_10hz\s*=\s*(?P<name>\w+)\[\"pred_traj_10hz\"\]", body)
        if rollout_match is None:
            raise RuntimeError("Could not infer rollout output variable from committed path builder call.")
        rollout_var = rollout_match.group("name")
        indent = match.group("indent")
        prep = (
            f"{indent}if self.self_forced_delayed_window_enabled:\n"
            f"{indent}    tokenized_agent = build_delayed_anchor0_tokenized_agent(\n"
            f"{indent}        tokenized_agent=tokenized_agent,\n"
            f"{indent}        pred_traj_10hz={rollout_var}[\"pred_traj_10hz\"],\n"
            f"{indent}        pred_head_10hz={rollout_var}[\"pred_head_10hz\"],\n"
            f"{indent}        window=delayed_window,\n"
            f"{indent}        commit_steps=int(self.encoder.agent_encoder.shift),\n"
            f"{indent}    )\n"
        )
        text = text[: match.start()] + prep + text[match.start():]

    text = text.replace(
        "build_anchor0_normalized_committed_path(",
        "build_delayed_normalized_committed_path(",
    )

    write_text_if_changed(path, text)


def patch_smart_flow_decoder_py(repo_root: Path) -> None:
    """SMARTFlowDecoder wrapper에 학습 시작 block 인자를 전달합니다.

    Args:
        repo_root: CAT-K 저장소 루트입니다.

    Returns:
        None
    """
    path = repo_root / "src/smart/modules/smart_flow_decoder.py"
    text = read_text(path)
    if "learning_start_step_2hz" not in text:
        text = replace_once(
            text=text,
            old=(
                "        rollout_steps_2hz: int | None = None,\n"
                "        self_forced_epoch: int | None = None,\n"
            ),
            new=(
                "        rollout_steps_2hz: int | None = None,\n"
                "        learning_start_step_2hz: int = 0,\n"
                "        self_forced_epoch: int | None = None,\n"
            ),
            label="smart_flow_decoder training_rollout signature",
        )
        text = replace_once(
            text=text,
            old=(
                "            rollout_steps_2hz=rollout_steps_2hz,\n"
                "            self_forced_epoch=self_forced_epoch,\n"
            ),
            new=(
                "            rollout_steps_2hz=rollout_steps_2hz,\n"
                "            learning_start_step_2hz=learning_start_step_2hz,\n"
                "            self_forced_epoch=self_forced_epoch,\n"
            ),
            label="smart_flow_decoder pass delayed start",
        )
    write_text_if_changed(path, text)


def patch_flow_agent_decoder_py(repo_root: Path) -> None:
    """Flow agent rollout에 앞구간 detach 지점을 추가합니다.

    Args:
        repo_root: CAT-K 저장소 루트입니다.

    Returns:
        None
    """
    path = repo_root / "src/smart/modules/flow_agent_decoder.py"
    text = read_text(path)
    if "learning_start_step_2hz" not in text:
        old_signature_pair = (
            "        rollout_steps_2hz: int | None = None,\n"
            "        self_forced_epoch: int | None = None,\n"
        )
        new_signature_pair = (
            "        rollout_steps_2hz: int | None = None,\n"
            "        learning_start_step_2hz: int = 0,\n"
            "        self_forced_epoch: int | None = None,\n"
        )
        text = replace_exact_count(
            text=text,
            old=old_signature_pair,
            new=new_signature_pair,
            expected_count=2,
            label="flow_agent_decoder training_rollout signature",
        )
        loop_marker = "        for t in range(n_step_future_2hz):\n"
        text = insert_once_before(
            text=text,
            marker=loop_marker,
            insert="        learning_start_step_2hz = max(0, int(learning_start_step_2hz))\n",
            label="flow_agent_decoder delayed start normalize",
        )
        text = replace_once(
            text=text,
            old="            if detach_block_transition and t > 0:\n",
            new=(
                "            if (detach_block_transition and t > 0) or (\n"
                "                t == learning_start_step_2hz and t > 0\n"
                "            ):\n"
            ),
            label="flow_agent_decoder delayed prefix detach",
        )
        text = replace_once(
            text=text,
            old=(
                "            rollout_steps_2hz=rollout_steps_2hz,\n"
                "            self_forced_epoch=self_forced_epoch,\n"
            ),
            new=(
                "            rollout_steps_2hz=rollout_steps_2hz,\n"
                "            learning_start_step_2hz=learning_start_step_2hz,\n"
                "            self_forced_epoch=self_forced_epoch,\n"
            ),
            label="flow_agent_decoder pass delayed start",
        )
    write_text_if_changed(path, text)


def patch_readme(repo_root: Path) -> None:
    """README.md에 delayed-window self-forcing 사용법을 추가합니다.

    Args:
        repo_root: CAT-K 저장소 루트입니다.

    Returns:
        None
    """
    path = repo_root / "README.md"
    text = read_text(path)
    if README_MARKER not in text:
        text = text.rstrip() + README_SECTION + "\n"
    write_text_if_changed(path, text)


def assert_required_overlay_files(repo_root: Path, relative_paths: Iterable[str]) -> None:
    """zip에 들어 있어야 하는 새 파일들이 실제로 있는지 확인합니다.

    Args:
        repo_root: CAT-K 저장소 루트입니다.
        relative_paths: 확인할 상대 경로 목록입니다.

    Returns:
        None

    Raises:
        FileNotFoundError: 필요한 파일이 없는 경우입니다.
    """
    missing = [path for path in relative_paths if not (repo_root / path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The zip overlay files are missing. Unzip the package at the repo root first.\n"
            f"Missing files:\n{formatted}"
        )


def main() -> None:
    """패키지의 delayed-window self-forcing 변경을 현재 저장소에 적용합니다.

    Returns:
        None
    """
    repo_root = find_repo_root(Path.cwd())
    assert_required_overlay_files(
        repo_root=repo_root,
        relative_paths=[
            "src/smart/modules/self_forced_delayed_window.py",
            "configs/experiment/self_forced_delayed_npfm_h100_4.yaml",
            "configs/experiment/self_forced_delayed_npfm_h100_6.yaml",
            "tests/test_self_forced_delayed_window.py",
            "docs/DELAYED_SELF_FORCING_USAGE.md",
        ],
    )
    patch_smart_flow_py(repo_root)
    patch_smart_flow_decoder_py(repo_root)
    patch_flow_agent_decoder_py(repo_root)
    patch_readme(repo_root)
    print("Delayed-window self-forcing patch applied.")


if __name__ == "__main__":
    main()
