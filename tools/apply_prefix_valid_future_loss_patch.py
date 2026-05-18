from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
FLOW_TOKEN_PROCESSOR_PATH = ROOT_DIR / "src/smart/tokens/flow_token_processor.py"
SMART_FLOW_CONFIG_PATH = ROOT_DIR / "configs/model/smart_flow.yaml"
README_PATH = ROOT_DIR / "README.md"


def _read_text(path: Path) -> str:
    """파일 내용을 읽습니다.

    Args:
        path: 읽을 파일 경로입니다.

    Returns:
        str: 파일의 전체 문자열입니다.
    """
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    """파일 내용을 저장합니다.

    Args:
        path: 저장할 파일 경로입니다.
        text: 저장할 전체 문자열입니다.

    Returns:
        None
    """
    path.write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, file_path: Path) -> str:
    """문자열 조각을 한 번만 바꿉니다.

    Args:
        text: 원본 파일 내용입니다.
        old: 바꿀 기존 문자열입니다.
        new: 새 문자열입니다.
        file_path: 오류 메시지에 표시할 파일 경로입니다.

    Returns:
        str: 변경된 파일 내용입니다.

    Raises:
        RuntimeError: 기존 문자열을 찾지 못한 경우입니다.
    """
    if old not in text:
        raise RuntimeError(f"Cannot find expected block in {file_path}")
    return text.replace(old, new, 1)


def _patch_flow_token_processor() -> None:
    """FlowTokenProcessor에 prefix-valid 미래 loss mask 옵션을 추가합니다.

    Args:
        없음.

    Returns:
        None
    """
    text = _read_text(FLOW_TOKEN_PROCESSOR_PATH)

    if "use_prefix_valid_future_loss_mask: bool = False" not in text:
        text = _replace_once(
            text=text,
            old="""        flow_window_steps: int = 20,\n    ) -> None:\n""",
            new="""        flow_window_steps: int = 20,\n        use_prefix_valid_future_loss_mask: bool = False,\n    ) -> None:\n""",
            file_path=FLOW_TOKEN_PROCESSOR_PATH,
        )

    if "self.use_prefix_valid_future_loss_mask = bool(use_prefix_valid_future_loss_mask)" not in text:
        text = _replace_once(
            text=text,
            old="""        self.flow_window_steps = validate_flow_window_steps(\n            flow_window_steps=flow_window_steps,\n            commit_steps=self.shift,\n        )\n""",
            new="""        self.flow_window_steps = validate_flow_window_steps(\n            flow_window_steps=flow_window_steps,\n            commit_steps=self.shift,\n        )\n        self.use_prefix_valid_future_loss_mask = bool(use_prefix_valid_future_loss_mask)\n""",
            file_path=FLOW_TOKEN_PROCESSOR_PATH,
        )

    if "def _build_prefix_valid_future_loss_mask" not in text:
        old_block = """    def _build_anchor_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:\n        \"\"\"현재 anchor 뒤 전체 flow window가 유효한 경우에만 미래 mask를 만듭니다.\n\n        Args:\n            valid: 각 agent와 시점의 유효 여부입니다.\n                shape은 ``[n_agent, n_step]`` 입니다.\n            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.\n\n        Returns:\n            Tensor:\n                미래 step별 loss 사용 여부입니다.\n                shape은 ``[n_agent, flow_window_steps]`` 입니다.\n                전체 미래 window가 유효한 agent만 모든 step이 ``True`` 입니다.\n        \"\"\"\n        future_start = raw_step + 1\n        # future_mask: [n_agent, flow_window_steps]\n        future_mask = torch.zeros(\n            (valid.shape[0], self.flow_window_steps),\n            device=valid.device,\n            dtype=torch.bool,\n        )\n        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))\n        if available_len <= 0:\n            return future_mask\n\n        # available_future_valid: [n_agent, available_len]\n        available_future_valid = valid[:, future_start : future_start + available_len].bool()\n        if available_len != self.flow_window_steps:\n            return future_mask\n\n        full_future_valid = available_future_valid.all(dim=1)\n        future_mask[full_future_valid] = True\n        return future_mask\n"""
        new_block = """    def _build_anchor_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:\n        \"\"\"현재 설정에 맞는 미래 loss mask를 만듭니다.\n\n        Args:\n            valid: 각 agent와 시점의 유효 여부입니다.\n                shape은 ``[n_agent, n_step]`` 입니다.\n            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.\n\n        Returns:\n            Tensor:\n                미래 step별 loss 사용 여부입니다.\n                shape은 ``[n_agent, flow_window_steps]`` 입니다.\n        \"\"\"\n        if self.use_prefix_valid_future_loss_mask:\n            return self._build_prefix_valid_future_loss_mask(valid=valid, raw_step=raw_step)\n        return self._build_full_window_future_loss_mask(valid=valid, raw_step=raw_step)\n\n    def _build_full_window_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:\n        \"\"\"기존 방식처럼 전체 미래 window가 유효한 경우에만 loss mask를 만듭니다.\n\n        Args:\n            valid: 각 agent와 시점의 유효 여부입니다.\n                shape은 ``[n_agent, n_step]`` 입니다.\n            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.\n\n        Returns:\n            Tensor:\n                미래 step별 loss 사용 여부입니다.\n                shape은 ``[n_agent, flow_window_steps]`` 입니다.\n                미래 전체가 유효한 agent만 모든 step이 ``True`` 입니다.\n        \"\"\"\n        future_start = raw_step + 1\n        # future_loss_mask: [n_agent, flow_window_steps]\n        future_loss_mask = torch.zeros(\n            (valid.shape[0], self.flow_window_steps),\n            device=valid.device,\n            dtype=torch.bool,\n        )\n        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))\n        if available_len != self.flow_window_steps:\n            return future_loss_mask\n\n        # available_future_valid: [n_agent, flow_window_steps]\n        available_future_valid = valid[:, future_start : future_start + available_len].bool()\n        full_future_valid = available_future_valid.all(dim=1)\n        future_loss_mask[full_future_valid] = True\n        return future_loss_mask\n\n    def _build_prefix_valid_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:\n        \"\"\"가까운 미래부터 연속으로 유효한 구간만 loss mask로 만듭니다.\n\n        Args:\n            valid: 각 agent와 시점의 유효 여부입니다.\n                shape은 ``[n_agent, n_step]`` 입니다.\n            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.\n\n        Returns:\n            Tensor:\n                미래 step별 loss 사용 여부입니다.\n                shape은 ``[n_agent, flow_window_steps]`` 입니다.\n                ``raw_step + 1``부터 처음 유효하지 않은 step 직전까지만\n                ``True`` 입니다. 첫 미래 step이 유효하지 않으면 전부 ``False`` 입니다.\n        \"\"\"\n        future_start = raw_step + 1\n        # future_loss_mask: [n_agent, flow_window_steps]\n        future_loss_mask = torch.zeros(\n            (valid.shape[0], self.flow_window_steps),\n            device=valid.device,\n            dtype=torch.bool,\n        )\n        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))\n        if available_len <= 0:\n            return future_loss_mask\n\n        # available_future_valid: [n_agent, available_len]\n        available_future_valid = valid[:, future_start : future_start + available_len].bool()\n        # prefix_valid: [n_agent, available_len]\n        prefix_valid = available_future_valid.to(dtype=torch.long).cumprod(dim=1).bool()\n        future_loss_mask[:, :available_len] = prefix_valid\n        return future_loss_mask\n"""
        text = _replace_once(
            text=text,
            old=old_block,
            new=new_block,
            file_path=FLOW_TOKEN_PROCESSOR_PATH,
        )

    _write_text(FLOW_TOKEN_PROCESSOR_PATH, text)


def _patch_smart_flow_config() -> None:
    """기본 model config에 prefix-valid 옵션을 추가합니다.

    Args:
        없음.

    Returns:
        None
    """
    text = _read_text(SMART_FLOW_CONFIG_PATH)
    if "use_prefix_valid_future_loss_mask" in text:
        return

    text = _replace_once(
        text=text,
        old="""    flow_window_steps: ${model.model_config.decoder.flow_window_steps}\n""",
        new="""    flow_window_steps: ${model.model_config.decoder.flow_window_steps}\n    # false: 기존 방식. flow_window_steps 전체 미래가 유효한 anchor만 학습합니다.\n    # true: 가장 가까운 미래부터 연속으로 유효한 prefix만 loss를 주고 학습합니다.\n    use_prefix_valid_future_loss_mask: true\n""",
        file_path=SMART_FLOW_CONFIG_PATH,
    )
    _write_text(SMART_FLOW_CONFIG_PATH, text)


def _build_readme_section() -> str:
    """README에 넣을 prefix-valid 사용법 섹션을 만듭니다.

    Args:
        없음.

    Returns:
        str: README.md에 삽입할 섹션입니다.
    """
    return """### 5.1.2 미래 GT 유효 길이 기반 학습 target 선택\n\n학습 target 선택은 아래 단일 옵션으로 고릅니다.\n\n```bash\nmodel.model_config.token_processor.use_prefix_valid_future_loss_mask=false  # 기존 방식\nmodel.model_config.token_processor.use_prefix_valid_future_loss_mask=true   # prefix-valid 방식\n```\n\n- `false`이면 기존과 같습니다. 현재 anchor 뒤 `decoder.flow_window_steps` 전체 미래가 모두 유효한 agent-anchor만 학습합니다.\n- `true`이면 현재 anchor 뒤 가장 가까운 미래부터 시작해서, 처음 끊기기 전까지 연속으로 유효한 구간만 학습합니다. 이 구간에만 loss가 들어갑니다.\n- full-valid sample은 `true`에서도 그대로 전체 미래 loss를 받습니다. 새로 추가되는 것은 partial-valid sample뿐입니다.\n- 이 옵션은 `FlowTokenProcessor`에서 학습 target을 만들 때 적용되므로 pretrain, 일반 fine tuning, self-forced fine tuning에서 같은 방식으로 동작합니다.\n- README 기준 cache를 그대로 만들었다면 cache 재생성은 필요 없습니다. pkl cache 자체에서 partial-valid agent/anchor를 직접 삭제한 경우에만 cache를 다시 만들어야 합니다.\n\n기존 pretrained checkpoint를 prefix-valid 목표로 이어서 학습할 때는 `action=finetune`을 씁니다. 이 방식은 모델 weight만 불러오고 optimizer / scheduler는 새로 시작합니다. 모델 전체를 학습하려면 `model.model_config.finetune.enabled=false`를 유지합니다.\n\n#### H100 4GPU 단일 pod prefix-valid fine tuning\n\n```bash\nCUDA_VISIBLE_DEVICES=0,1,2,3 \\\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\ntorchrun \\\n  --standalone \\\n  --nproc_per_node=4 \\\n  -m src.run \\\n  experiment=finetune_flow_prefix_valid_h100_4 \\\n  paths.cache_root=\"$CACHE_ROOT\" \\\n  ckpt_path=\"/path/to/pretrained.ckpt\" \\\n  task_name=flow_prefix_valid_finetune_h100_4\n```\n\n#### A100 4GPU x 2node prefix-valid fine tuning\n\n각 node에서 같은 command를 실행하되 `--node_rank`만 다르게 둡니다.\n\n```bash\n# node 0\nCUDA_VISIBLE_DEVICES=0,1,2,3 \\\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\ntorchrun \\\n  --nnodes=2 \\\n  --nproc_per_node=4 \\\n  --node_rank=0 \\\n  --master_addr=<node0-address> \\\n  --master_port=29500 \\\n  -m src.run \\\n  experiment=finetune_flow_prefix_valid_a100_4x2 \\\n  paths.cache_root=\"$CACHE_ROOT\" \\\n  ckpt_path=\"/path/to/pretrained.ckpt\" \\\n  task_name=flow_prefix_valid_finetune_a100_4x2\n\n# node 1\nCUDA_VISIBLE_DEVICES=0,1,2,3 \\\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\ntorchrun \\\n  --nnodes=2 \\\n  --nproc_per_node=4 \\\n  --node_rank=1 \\\n  --master_addr=<node0-address> \\\n  --master_port=29500 \\\n  -m src.run \\\n  experiment=finetune_flow_prefix_valid_a100_4x2 \\\n  paths.cache_root=\"$CACHE_ROOT\" \\\n  ckpt_path=\"/path/to/pretrained.ckpt\" \\\n  task_name=flow_prefix_valid_finetune_a100_4x2\n```\n\n두 preset 모두 `max_epochs=16`, `lr=1e-4`, `lr_warmup_steps=1`, `gradient_clip_val=1.0`, `val_open_loop=true`, `val_closed_loop=true`를 사용합니다. `decoder.flow_window_steps`는 checkpoint와 같은 값을 써야 합니다. 2초 pretrained checkpoint면 기본값 `20`을 그대로 둡니다.\n\n"""


def _patch_readme() -> None:
    """README의 학습 target 선택 설명을 prefix-valid 옵션 기준으로 갱신합니다.

    Args:
        없음.

    Returns:
        None
    """
    if not README_PATH.exists():
        return

    text = _read_text(README_PATH)
    if "experiment=finetune_flow_prefix_valid_h100_4" in text:
        return

    start_marker = "### 5.1.2 전체 유효 미래 window 학습 방식"
    end_marker = "### 5.2 Validation 주기와 val_open / val_closed 바꾸기"
    new_section = _build_readme_section()

    if start_marker in text and end_marker in text:
        start = text.index(start_marker)
        end = text.index(end_marker)
        text = text[:start] + new_section + text[end:]
    else:
        insertion_marker = "### 5.1 학습 설정을 거칠게 이해하는 법"
        if insertion_marker not in text:
            text = text.rstrip() + "\n\n" + new_section
        else:
            insert_at = text.index(insertion_marker)
            text = text[:insert_at] + new_section + text[insert_at:]

    _write_text(README_PATH, text)


def main() -> None:
    """repo root에서 prefix-valid future loss 변경을 적용합니다.

    Args:
        없음.

    Returns:
        None
    """
    _patch_flow_token_processor()
    _patch_smart_flow_config()
    _patch_readme()
    print("prefix-valid future loss patch applied successfully.")


if __name__ == "__main__":
    main()
