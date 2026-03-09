# CAT-K Flow Overlay for `seulbinHwang/catk:test`

이 overlay는 `catk/test`의 기존 Hydra 학습/검증/제출 흐름은 유지하고,
agent head만 **2.0초 sparse conditional flow matching**으로 바꾸는 패치입니다.

핵심 원칙
- **유지**: Waymo 데이터 파이프라인, token processor, map encoder, WOSAC 제출 흐름
- **교체**: next-token / GMM 중심 agent motion head
- **반드시 반영**:
  - 4-step midpoint ODE 적분
  - optional short closed-loop fine-tuning
  - `(sin, cos)` 재정규화로 heading drift 완화
- **반영하지 않음**:
  - warm-start inference
  - WaymoTargetBuilder의 무작위 32개 제한 제거
  - random anchor 1개 기본값

## 1. 덮어쓰기 적용

`catk/test` checkout 상태의 repo 루트에서 아래처럼 덮어씁니다.

```bash
cp -r /path/to/catk_flow_patch_final/* /path/to/catk/
```

## 2. 바뀐 파일

주요 변경 파일
- `src/smart/modules/agent_flow_decoder.py`
- `src/smart/modules/smart_decoder.py`
- `src/smart/model/smart.py`
- `src/smart/metrics/flow_loss.py`
- `src/smart/metrics/__init__.py`
- `src/smart/utils/flow_traj.py`
- `src/smart/utils/__init__.py`
- `src/smart/utils/finetune.py`
- `configs/model/flow_smart.yaml`
- `configs/experiment/flow_pretrain_h1006.yaml`
- `configs/experiment/flow_clsft_h1006.yaml`
- `configs/experiment/flow_local_val.yaml`
- `configs/experiment/flow_wosac_sub.yaml`
- `scripts/train_flow_h1006.sh`
- `scripts/finetune_flow_h1006.sh`
- `scripts/local_val_flow.sh`
- `scripts/wosac_sub_flow.sh`

## 3. 구조 요약

### 3.1 open-loop pretraining
- scene 내부 anchor 후보를 만듭니다.
- 한 step에서 `anchor_chunk_k=4`개만 랜덤으로 뽑습니다.
- 각 anchor마다 2.0초 GT future를 `[4, 6, 4]` 조각으로 바꿉니다.
- OT path로 `z_tau`를 만들고 velocity field를 학습합니다.
- 손실은
  - flow loss
  - overlap consistency loss
  입니다.

### 3.2 short closed-loop fine-tuning
- config에서 `closed_loop_steps=4`로 켜집니다.
- 0.5초씩 4번 굴립니다.
- 매 step마다 2.0초 future를 새로 샘플링합니다.
- 맨 앞 0.5초만 scene 상태 갱신에 사용합니다.
- 그 0.5초 continuous trajectory를 nearest SMART token으로 다시 바꿔 내부 history에 넣습니다.

### 3.3 inference / submission
- warm-start는 쓰지 않습니다.
- 매 0.5초마다 새 noise에서 2.0초 future를 생성합니다.
- 4-step midpoint ODE를 씁니다.
- 조립 뒤와 적분 중간마다 `(sin, cos)` 재정규화를 합니다.
- WOSAC 제출 형식은 기존 `catk/test` 경로를 그대로 씁니다.

## 4. H100 6장 기준 추천 설정

이 값은 `catk/test`의 `SMART-tiny-7M` 계열 설정을 최대한 유지하면서,
`A100 8장` 대신 `H100 6장`에 맞게 **보수적으로** 잡은 추천값입니다.

### pretrain
- hidden dim: `128`
- map layers: `3`
- agent layers: `6`
- future window: `20`
- anchor chunk: `4`
- batch per GPU: `12`
- global batch: `72`
- lr: `5e-4`
- epochs: `64`
- precision: `bf16-mixed`
- grad clip: `0.5`

### short closed-loop fine-tuning
- batch per GPU: `6`
- accumulate grad: `2`
- effective global batch: `72`
- lr: `5e-5`
- epochs: `16`
- closed_loop_steps: `4`
- precision: `bf16-mixed`

메모리가 남으면 pretrain batch를 `14`까지 올려볼 수 있지만,
첫 시작점은 위 설정을 권합니다.

## 5. 실행 순서

### 5.1 open-loop pretraining

```bash
bash scripts/train_flow_h1006.sh
```

직접 실행할 때는:

```bash
torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_pretrain_h1006 \
  trainer.devices=6 \
  task_name=flow_pretrain_h1006
```

### 5.2 optional short closed-loop fine-tuning

```bash
bash scripts/finetune_flow_h1006.sh /path/to/flow_pretrain.ckpt
```

직접 실행할 때는:

```bash
torchrun \
  --nproc_per_node=6 \
  -m src.run \
  experiment=flow_clsft_h1006 \
  ckpt_path=/path/to/flow_pretrain.ckpt \
  trainer.devices=6 \
  task_name=flow_clsft_h1006
```

### 5.3 local validation

```bash
bash scripts/local_val_flow.sh /path/to/flow_clsft.ckpt
```

### 5.4 WOSAC submission 생성

```bash
bash scripts/wosac_sub_flow.sh /path/to/flow_clsft.ckpt validate
```

또는

```bash
bash scripts/wosac_sub_flow.sh /path/to/flow_clsft.ckpt test
```

## 6. old NTP / GMM 코드 정리

이번 flow 경로만 남기고 싶으면 아래 파일 목록을 참고해서 지우면 됩니다.

- `DELETE_THESE_FILES.txt`
- `scripts/prune_legacy_ntp_gmm.sh`

주의:
- `src/smart/modules/agent_decoder.py`는 **지우면 안 됩니다.**
- flow decoder가 기존 sparse edge builder를 재사용합니다.

## 7. 구현상 주의점

- `catk/test`의 WOSAC 입력은 관측 과거가 1초라서, rollout 시작 시점에 관측 history token은 2개뿐입니다.
- 대신 rollout이 진행될수록 예측 결과를 nearest SMART token으로 다시 넣어서, 내부 memory는 최대 6 slot까지 유지됩니다.
- 즉, **초기 관측 과거는 짧지만 내부 token state space 자체는 6-slot rolling memory로 유지**됩니다.

## 8. 꼭 확인할 것

- 토큰 파일 경로
  - `map_traj_token5.pkl`
  - `agent_vocab_555_s2.pkl`
- Waymo cache 경로
- 제출 metadata
  - `account_name`
  - `affiliation`
  - `method_link`

