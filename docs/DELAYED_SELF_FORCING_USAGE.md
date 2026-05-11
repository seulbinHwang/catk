# Delayed-Window Self-Forcing 사용법

이 패치는 self-forcing fine-tuning에서 학습 시작 시점만 순차적으로 뒤로 미룹니다.
RMM을 직접 loss로 쓰지 않습니다. 새 random window도 쓰지 않습니다.

## 학습 schedule

| epoch | 전체 rollout | 학습하지 않는 구간 | 실제 학습 구간 | 기준 시점 |
|---:|---:|---:|---:|---:|
| 0~3 | 2초 | 없음 | 0~2초 | 0초 |
| 4~7 | 4초 | 0~2초 | 2~4초 | 2초 |
| 8~11 | 6초 | 0~4초 | 4~6초 | 4초 |
| 12~15 | 8초 | 0~6초 | 6~8초 | 6초 |

핵심 규칙은 다음과 같습니다.

1. 앞구간은 현재 모델이 스스로 굴러가게만 합니다.
2. 앞구간은 loss에 쓰지 않습니다.
3. 앞구간과 실제 학습 구간 사이의 gradient 연결은 끊습니다.
4. 실제 학습 구간 안의 0.5초 block 연결은 유지합니다.
5. 항상 2초 target만 학습합니다.

## 코드 반영 방법

`self_forcing_2` 브랜치 루트에서 zip을 풀고 아래 명령을 실행합니다.

```bash
python tools/apply_delayed_self_forcing_patch.py
```

스크립트는 다음 파일을 수정합니다.

- `src/smart/model/smart_flow.py`
- `src/smart/modules/smart_flow_decoder.py`
- `src/smart/modules/flow_agent_decoder.py`
- `README.md`

그리고 다음 파일은 zip에 포함되어 그대로 복사됩니다.

- `src/smart/modules/self_forced_delayed_window.py`
- `configs/experiment/self_forced_delayed_npfm_h100_4.yaml`
- `configs/experiment/self_forced_delayed_npfm_h100_6.yaml`
- `tests/test_self_forced_delayed_window.py`

## 6x H100 실행 예시

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=6 -m src.run \
  experiment=self_forced_delayed_npfm_h100_6 \
  action=finetune \
  paths.cache_root="$CACHE_ROOT" \
  task_name=self_forced_delayed_window_h100_6 \
  ckpt_path="/path/to/pretrained.ckpt"
```

## 4x H100 실행 예시

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 -m src.run \
  experiment=self_forced_delayed_npfm_h100_4 \
  action=finetune \
  paths.cache_root="$CACHE_ROOT" \
  task_name=self_forced_delayed_window_h100_4 \
  ckpt_path="/path/to/pretrained.ckpt"
```

## 고정된 설정

| 항목 | 값 |
|---|---:|
| target horizon | 2초 |
| `decoder.flow_window_steps` | 20 |
| 전체 denoising | 32 step |
| 역전파 | 마지막 8 step |
| 학습 epoch | 16 |
| estimator 안정화 | 1 epoch |
| block transition detach | false |
| delayed 앞구간 detach | true |
| RMM 직접 loss | 사용 안 함 |

## 주의사항

- 이 학습법은 2초 target 전용입니다. `decoder.flow_window_steps=20`이 아니면 실행 중 명확히 실패하게 했습니다.
- `detach_block_transition=false`는 실제 학습 2초 구간 내부 연결을 유지하기 위한 설정입니다.
- 앞구간은 별도 조건으로 detach하므로, 2~4초/4~6초/6~8초 loss가 앞구간 실행 결과를 직접 바꾸지 않습니다.
