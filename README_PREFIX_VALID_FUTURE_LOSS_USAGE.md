# Prefix-valid Future Loss Patch

이 zip은 `self_forcing_w_track_loss` 브랜치용 변경 묶음입니다.

## 적용 방법

repo root에서 압축을 풀고 아래를 한 번 실행하세요.

```bash
python tools/apply_prefix_valid_future_loss_patch.py
```

적용되는 변경은 다음입니다.

1. `src/smart/tokens/flow_token_processor.py`
   - `use_prefix_valid_future_loss_mask` 옵션 추가
   - `false`: 기존처럼 `decoder.flow_window_steps` 전체 미래가 유효한 anchor만 학습
   - `true`: 가장 가까운 미래부터 연속 유효한 prefix만 학습하고 해당 prefix에만 loss 적용

2. `configs/model/smart_flow.yaml`
   - 기본값 추가

```yaml
model:
  model_config:
    token_processor:
      use_prefix_valid_future_loss_mask: true
```

3. `README.md`
   - prefix-valid target 선택 방식과 fine tuning 사용법 자동 반영

4. 새 experiment config 추가
   - `configs/experiment/finetune_flow_prefix_valid_h100_4.yaml`
   - `configs/experiment/finetune_flow_prefix_valid_a100_4x2.yaml`

## H100 4GPU 단일 pod 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_h100_4 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_h100_4
```

## A100 4GPU x 2node 실행

node 0:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=<node0-address> \
  --master_port=29500 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_a100_4x2 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_a100_4x2
```

node 1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=1 \
  --master_addr=<node0-address> \
  --master_port=29500 \
  -m src.run \
  experiment=finetune_flow_prefix_valid_a100_4x2 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="/path/to/pretrained.ckpt" \
  task_name=flow_prefix_valid_finetune_a100_4x2
```

## 모든 학습에 적용하는 방법

어떤 experiment든 아래 override 하나만 추가하면 됩니다.

```bash
model.model_config.token_processor.use_prefix_valid_future_loss_mask=true
```

기존 방식으로 되돌리려면 아래처럼 둡니다.

```bash
model.model_config.token_processor.use_prefix_valid_future_loss_mask=false
```

## cache 관련

README 방식으로 만든 cache라면 재생성하지 않아도 됩니다. pkl cache 자체에서 partial-valid agent나 anchor를 직접 삭제한 경우에만 cache를 다시 만드세요.
