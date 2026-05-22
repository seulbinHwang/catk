# 토큰화 교통 모델의 폐루프 지도 미세조정

<p align="center">
     <img src="docs/catk_banner.png" alt="Top-K 중 가장 가까운 선택(CAT-K) rollout은 미세조정 중 정책을 unroll할 때 방문 상태가 정답 궤적에 가깝게 유지되도록 만든다.", width=760px>
     <br/><strong>Top-K 중 가장 가까운 선택(CAT-K) Rollout</strong>은 미세조정 중 정책을 unroll할 때 방문 상태가 정답 궤적(GT)에 가깝게 유지되도록 만든다. 각 시점에서 CAT-K는 먼저 정책이 가장 그럴듯하다고 보는 top-K action token을 뽑고, 그중 다음 상태가 GT와 가장 가까운 token을 선택한다. 그래서 CAT-K rollout은 GT의 mode, 예를 들어 좌회전 궤적을 따라가며, random rollout이나 단순 top-K rollout처럼 직진/우회전으로 크게 벗어나는 일을 줄인다. 정책은 rollout 상태와 GT 상태 사이의 거리를 줄이도록 학습되므로, GT 기반 supervision이 CAT-K rollout에는 유효하게 작동하지만 random/top-K rollout에는 상대적으로 덜 유효하다.
</p>

> **토큰화 교통 모델의 폐루프 지도 미세조정**
> [Zhejun Zhang](https://zhejz.github.io/), [Peter Karkus](https://karkus.tilda.ws/), [Maximilian Igl](https://maximilianigl.com/), [Wenhao Ding](https://wenhao.pub/), [Yuxiao Chen](https://research.nvidia.com/labs/avg/author/yuxiao-chen/), [Boris Ivanovic](https://www.borisivanovic.com/), [Marco Pavone](https://web.stanford.edu/~pavone/index.html)<br/>
> 
> [프로젝트 페이지](https://zhejz.github.io/catk)<br/>
> [arXiv 논문](https://arxiv.org/abs/2412.05334)

아래 BibTeX 항목은 인용 정확성을 위해 원문 표기를 유지한다.

```bibtex
@inproceedings{zhang2025closed,
  title = {Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models},
  author = {Zhang, Zhejun and Karkus, Peter and Igl, Maximilian and Ding, Wenhao and Chen, Yuxiao and Ivanovic, Boris and Pavone, Marco},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
}
```

## 소식과 업데이트

2025년 4월
- **CVPR 2025 구두 발표 선정**: 기쁜 소식이다.
- **WOSAC 2024 리더보드 1위**: Waymo Challenges 2025가 다가오면서 WOSAC 2024 리더보드는 종료되었고, 이 방법은 최종 1위를 유지했다.

2025년 2월
- **논문 CVPR 2025 채택**: 기쁜 소식이다.

- **WOSAC용 모델 checkpoint**: WOSAC submission에 사용한 checkpoint(SMART-tiny-CLSFT)는 Zhejun에게 이메일(zhejun.zhang94@gmail.com)을 보내 받을 수 있다. Waymo 약관에 따라, [My Submissions](https://waymo.com/open/challenges/submissions) 페이지에 등록 및 로그인되어 있음을 보여주는 스크린샷을 함께 보내야 한다.

- **SMART-mini와 SMART-nano**: 7M parameter의 SMART-tiny는 8장 A100에서 며칠간 학습해야 하므로 비용 부담이 클 수 있다. 이를 위해 더 작은 모델 설정인 [smart_mini_3M.yaml](configs/model/smart_mini_3M.yaml)과 [smart_nano_1M.yaml](configs/model/smart_nano_1M.yaml)을 추가했다. 특히 SMART-nano-1M은 A100 한 장에서도 학습할 수 있지만 성능은 상당히 낮다. Pre-training과 CAT-K fine-tuning 이후 SMART-nano-1M은 RMM 0.74를 달성했으며, 이는 SMART-tiny-7M보다 0.03 낮다.

2025년 1월
- **WOSAC 최고 수준 성능**: CAT-K는 [WOSAC 리더보드](https://waymo.com/open/challenges/2024/sim-agents/) 1위를 달성했다. agent token vocabulary 문제를 해결한 뒤 fine-tuned model은 **0.7702** RMM을 달성했다. 리더보드에는 공개하지 않았지만 BC로 32 epoch만 학습한 재현 SMART-tiny-7M도 **0.7671** RMM을 달성했고, 이는 당시 2위 방법과 비슷한 수준이다. 재현 절차도 비교적 단순하다.

- **agent token vocabulary 문제**: 기존에 사용하던 [agent token vocabulary 파일](src/smart/tokens/cluster_frame_5_2048_remove_duplicate.pkl)은 [SMART 저장소](https://github.com/rainmaker22/SMART/blob/main/smart/tokens/cluster_frame_5_2048.pkl)에서 가져온 것이지만, 최적 성능 재현용이 아니라 sanity check 용도였다. 이를 해결하기 위해 [trajectory clustering 스크립트](src/smart/tokens/traj_clustering.py)를 추가하고 [적절한 agent token vocabulary](src/smart/tokens/agent_vocab_555_s2.pkl)를 새로 만들었다. 이 스크립트는 SMART의 [k-disk clustering 스크립트](https://github.com/rainmaker22/SMART/blob/main/scripts/traj_clstering.py)를 기반으로 한다. 업데이트된 agent token 덕분에 모든 traffic simulation model에서 RMM이 약 +0.0060 개선되었다.

## 설치

- 가장 간단한 환경 구성 방법은 [conda](https://docs.conda.io/en/latest/miniconda.html) 환경을 만들고 아래 명령을 실행하는 것이다.

  ```bash
  conda create -y -n catk python=3.11.9
  conda activate catk
  conda install -y -c conda-forge ffmpeg=4.3.2
  pip install -r install/requirements.txt
  pip install torch_geometric
  pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
  ```

- 다른 방법으로 [Dockerfile](install/Dockerfile)을 사용해 직접 Docker image를 만들 수 있다. 몇 가지 이유로 Docker 환경에서 코드가 더 빠르게 실행되는 것을 확인했다.
- logging에는 [WandB](https://wandb.ai/)를 사용한다. 계정은 무료로 만들 수 있다.
  - `configs/logger/wandb.yaml`은 `semi_control_stable`과 같은 기본 계정 설정을
    사용한다. 기본값은 `project: SMART-FLOW`,
    `entity: jksg01019-naver-labs`, `log_model: all`이다. 따라서 Lightning
    `ModelCheckpoint`가 저장하는 checkpoint는 W&B model artifact로 업로드된다.
    별도 callback이 저장하는 `epoch_last.ckpt`도 W&B artifact로 업로드된다.
    다른 project/entity를 쓰려면 실행 시 `logger.wandb.project=<project>` 또는
    `logger.wandb.entity=<entity>`를 override하면 된다. 로컬 디스크에만 기록하고
    싶으면 환경변수 `WANDB_MODE=offline`을 켜거나 `logger.wandb.offline=true`를
    추가하면 wandb 서버로의 업로드가 멈춘다. 이때 Lightning은 offline mode에서
    model artifact 업로드(`log_model: all`)를 허용하지 않으므로, logger 생성 직전에
    `log_model=false`로 자동 낮춘다. 따라서 online 실행에서는 checkpoint artifact
    업로드 기본값을 유지하고, offline smoke test나 로컬 디버깅은 별도
    `logger.wandb.log_model=false` override 없이 실행할 수 있다.
- **주의할 점**
  - 학습과 validation에는 *NVIDIA A100 80GB* 8장을 사용했다. 학습과 fine-tuning은 며칠이 걸리고, validation과 test도 몇 시간이 걸릴 수 있다.
  - [Waymo Open Motion Dataset 약관](https://waymo.com/open/terms)에 따라 pre-trained model은 공유할 수 없다.

## 데이터셋 준비

- [Waymo Open Motion Dataset](https://waymo.com/open/download/)을 다운로드한다. 이 저장소는 v1.2.1을 사용한다.
- [scripts/cache_womd.sh](scripts/cache_womd.sh)를 사용해 dataset을 pickle 파일로 전처리하면 학습과 평가 중 data loading을 빠르게 할 수 있다.
- `training`, `validation`, `testing` 세 split을 모두 cache로 만들어야 한다.
- cache 생성 시 `crosswalk`, `speed_bump`, `driveway`는 서로 다른 map surface category로 보존된다. Point type은 각각 `9`, `10`, `11`을 사용하고, polygon category embedding도 `lane`, `road_edge`, `road_line`, `crosswalk`, `speed_bump`, `driveway` 6종을 입력받는다. 예전 cache처럼 `speed_bump`/`driveway`가 `crosswalk`로 합쳐져 있으면 네트워크가 세 surface를 구분할 수 없으므로, 이 구분을 쓰려면 cache를 새로 생성해야 한다.

## 코드 실행

제공하는 script는 다음과 같다.

- [scripts/train.sh](scripts/train.sh): 학습과 fine-tuning 실행
- [scripts/local_val.sh](scripts/local_val.sh): local validation 실행
- [scripts/wosac_sub.sh](scripts/wosac_sub.sh): submission 파일 패키징

기본 script는 single GPU로 실행된다. Multi-GPU 학습과 validation에는 DDP를 사용하며, 관련 실행 예시는 bash script 안에 들어 있다.

최종 결과를 재현하려면 아래 절차를 따르면 된다.

1. [BC pre-training config](configs/experiment/pre_bc.yaml)와 [scripts/train.sh](scripts/train.sh)를 사용해 SMART-tiny 7M 모델을 pre-train한다.
2. 1단계에서 pre-train한 SMART-tiny 모델을 [CLSFT with CAT-K config](configs/experiment/clsft.yaml)와 [scripts/train.sh](scripts/train.sh)로 fine-tune한다.
3. [scripts/wosac_sub.sh](scripts/wosac_sub.sh)를 사용해 `validate` 또는 `test` split용 submission 파일을 만든다. `logs` 폴더의 `sim_agents_2025_submission.tar.gz`를 [2025 Sim Agents 리더보드](https://waymo.com/open/challenges/2025/sim-agents/)에 업로드하면 2단계에서 fine-tune한 모델을 리더보드에서 평가할 수 있다.
4. 또는 [scripts/local_val.sh](scripts/local_val.sh)로 local validation을 실행할 수 있다.

### 공정 비교용 SMART NTP 학습 대상 선택

KFM 계열 실험과 SMART NTP pretrain을 공정하게 비교할 때는 학습 손실을 받는
agent 집합도 명시해야 한다. 현재 main 브랜치의 SMART NTP pretrain 기본값은
`configs/data/waymo.yaml`, `configs/experiment/pre_bc.yaml`,
`configs/experiment/pre_bc_a100x4x2.yaml` 모두에서
`data.train_use_eval_agent_selection: true`를 사용한다.

이 설정에서는 validation과 같은 agent selection을 학습에도 사용한다. 학습 전용
`train_mask`를 만들지 않으므로, SMART NTP loss는 별도 `train_mask` cap 없이
valid agent/time 전체를 기준으로 계산된다. 따라서 `data.train_max_num`은 이 경로의
학습 target 수를 제한하지 않는다.

legacy 학습 전용 target selection이 필요한 비교 실험에서는 실행 시
`data.train_use_eval_agent_selection=false`로 명시적으로 override해야 한다. 그 경우에만
ego 기준 150m 밖 agent valid mask를 자르고, role agent 또는 현재 100m 이내이면서 미래
valid가 충분한 agent 중 scenario당 최대 `data.train_max_num`개까지 `train_mask`로 골라
loss에 반영한다.

main 브랜치의 기본 trainer precision은 `configs/trainer/default.yaml`에서
`bf16-mixed`로 설정한다. 따라서 별도 precision override가 없는 SMART NTP pretrain,
local validation, WOSAC submission 생성은 모두 KFM 계열 `semi_control_stable` 설정과
같은 mixed bfloat16 실행 조건을 사용한다. A100/H100 같은 bf16 지원 GPU에서 학습과
추론의 precision 조건을 맞추기 위한 기본값이며, FP32가 꼭 필요한 실험만 실행 시
`trainer.precision=32-true`로 명시적으로 덮어쓴다.

bf16 mixed precision에서는 module 출력이 bfloat16으로 autocast될 수 있지만, cache에서
읽은 위치/heading tensor는 float32로 남는다. SMART agent decoder는 agent token
embedding을 임시 buffer에 모아 넣는 과정에서 buffer dtype을 token embedding 출력
dtype에 맞춘다. 그래서 `agent_token_embedding()`의 boolean indexing assignment가
`Float` destination과 `BFloat16` source mismatch로 실패하지 않고, pretrain,
local validation, WOSAC submission 경로가 기본 `bf16-mixed` 설정으로 실행된다.

### A100/H100x4x2 멀티 노드 SMART NTP pretrain

`testa`와 `testaa`처럼 A100 4장이 붙은 pod 두 개, 또는
`hsb-npc-training`과 `hsb-npc-training-2`처럼 H100 4장이 붙은 pod 두 개에서
SMART NTP pretrain을 바로 시작하려면 아래 wrapper를 사용한다.

```bash
bash scripts/start_smart_ntp_a100x4x2_testa_pretrain.sh
```

이 wrapper는 `scripts/launch_smart_ntp_a100x4x2_testa.py`를 호출하며, 기본
`TRAIN_BATCH_SIZE=12`와 task name
`smart_ntp_pretrain_a100x4x2_bs12_main`을 사용한다. 다른 batch를 실험하려면
`TRAIN_BATCH_SIZE=11 bash scripts/start_smart_ntp_a100x4x2_testa_pretrain.sh`
처럼 환경 변수로 넘긴다.

장기 학습 중 CUDA OOM이 나면 batch를 자동으로 낮춰 이어가야 하므로, 실험을 계속 살리는
목적의 wrapper도 별도로 둔다.

```bash
bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh
```

이 retry wrapper는 기본 `INITIAL_BS=12`, `OOM_STEP=1`, `MIN_BS=8`이다.
각 attempt는 같은 task name 아래에서 시작하고, pod tmux 로그에
`CUDA out of memory` / `torch.OutOfMemoryError` 같은 OOM marker가 보이면 두 pod의
학습 session을 정리한 뒤 `logs/<task_name>/runs/*/checkpoints/epoch_last.ckpt` 중
가장 최신 checkpoint를 찾아 다음 attempt의 `ckpt_path`로 넘긴다. 즉 OOM 이후에는
`data.train_batch_size`만 1 줄이고 optimizer, scheduler, epoch, global step을 가능한 한
보존해서 resume한다. checkpoint가 아직 없으면 같은 task name으로 새 attempt를 시작한다.
OOM이 아닌 실패는 조용히 batch를 낮추지 않고 중단한다.

retry wrapper는 기본적으로 pod 안의 `/tmp/catk_smart_ntp_a100x4x2_oom_retry_main`에
script-managed clean checkout을 준비하고, 매 attempt 전에 그 checkout을 `origin/main`으로
맞춘다. 따라서 기존 `/mnt/nuplan/projects/catk` checkout에 로컬 수정이나 detached HEAD가
남아 있어도 retry 학습 실행에는 영향을 주지 않는다. 다른 위치를 쓰려면
`PROJECT_ROOT=/path/to/checkout`을 명시한다.

주요 override는 환경 변수로 지정한다.

```bash
INITIAL_BS=12 MIN_BS=10 TASK_NAME=smart_ntp_pretrain_a100x4x2_retry_main \
  bash scripts/start_smart_ntp_a100x4x2_testa_pretrain_with_oom_retry.sh
```

launcher는 pod를 만들거나 지우지 않는다. 로컬에서 `kubectl exec`로 이미 떠 있는 두 pod에
접속한 뒤, 각 pod 안에서 같은 이름의 tmux session을 시작한다. 기본 namespace는 `p-pnc`,
기본 pod 목록은 `testa testaa`, branch는 `main`이다. 일반 launcher의 기본 원격 저장소
위치는 `/mnt/nuplan/projects/catk`이며, retry wrapper는 위에서 설명한 `/tmp` clean checkout을
기본으로 사용한다. 실행 전에 각 pod에서 대상 checkout을 현재 main에 맞춰 학습한다.
H100 pod에서 실행하려면 `--pods hsb-npc-training hsb-npc-training-2`를 명시한다.

#### H100 4+2 heterogeneous SMART NTP pretrain

`hsb-npc-training`의 H100 4장과 `wo-pvc`의 H100 2장을 묶어 총 6 rank로 같은
SMART NTP pretrain을 실행하려면 아래 wrapper를 사용한다.

```bash
bash scripts/start_smart_ntp_h100x4_h100x2_pretrain.sh
```

이 wrapper는 `scripts/launch_smart_ntp_h100x4_h100x2.py`를 호출하며, 기본 task name은
`smart_ntp_pretrain_h100x4_h100x2_bs13_main`이다. 기본 cache root는 두 pod 모두
`/workspace/womd_v1_3/SMART_cache`이고, 기본 experiment는 기존 A100x4x2와 같은
`pre_bc_a100x4x2`이다. 즉 SMART NTP 모델, tokenization, loss, validation scorer,
memory-balanced sampler 설정은 유지하고, 실행 pod/GPU 배치만 `4 + 2`로 바꾼다.

두 pod의 local GPU 수가 다르기 때문에 homogeneous `torchrun --nproc_per_node=4`를
쓰지 않는다. launcher가 각 pod의 GPU 수를 읽어 `hsb-npc-training`에는 rank `0~3`,
`wo-pvc`에는 rank `4~5`를 직접 배정하고,
`HeterogeneousTorchElasticEnvironment` / `HeterogeneousDDPStrategy`를 Hydra override로
넣어 Lightning의 `devices * num_nodes == WORLD_SIZE` 가정을 완화한다. sampler와
validation sharding은 launcher가 export한 실제 `WORLD_SIZE=6` 기준으로 동작한다.

기본값 기준 주요 학습 설정은 아래와 같다.

| 항목 | 값 |
|---|---|
| pod / GPU | `hsb-npc-training` 4GPU + `wo-pvc` 2GPU |
| total DDP ranks | 6 |
| experiment | `pre_bc_a100x4x2` |
| per-rank batch | `data.train_batch_size=13` |
| effective global batch | 78 |
| precision | `bf16-mixed` |
| lr / warmup / min ratio | `6e-4` / `4` / `1e-2` |
| validation | `scorer_scene_num=1680`, `check_val_every_n_epoch=16` |
| graph attention dtype | `CATK_ATTENTION_GRAPH_FP32=1` |

같은 per-rank batch를 바꾸려면 A100 wrapper와 동일하게 환경 변수로 넘긴다.

```bash
TRAIN_BATCH_SIZE=14 bash scripts/start_smart_ntp_h100x4_h100x2_pretrain.sh
```

실행 중 tmux에 붙으려면 아래 명령을 사용한다.

```bash
kubectl exec -it -n p-pnc hsb-npc-training -c main -- tmux attach -t catk-smart-ntp-h100x4-h100x2
kubectl exec -it -n p-pnc wo-pvc -c main -- tmux attach -t catk-smart-ntp-h100x4-h100x2
```

중단은 같은 task name으로 stop을 호출한다.

```bash
python scripts/launch_smart_ntp_h100x4_h100x2.py \
  --stop \
  --task-name smart_ntp_pretrain_h100x4_h100x2_bs13_main
```

기본 experiment는 `configs/experiment/pre_bc_a100x4x2.yaml`이다. 이 config는
`pre_bc`를 상속하므로 SMART backbone, next-token prediction loss,
deterministic nearest-token tokenization, agent selection, `num_freq_bands: 66`
같은 모델/알고리즘 설정은 바꾸지 않는다. 대신
`semi_control_stable`의 x4x2 control-space pretrain recipe와 학습 실행 조건을 맞추기
위해 아래 training/runtime 값만 명시한다.

- `trainer.devices: 4`, `trainer.num_nodes: 2`
- `data.train_batch_size: 12`, 즉 8개 rank 기준 effective global batch 96
- `model.model_config.lr: 6e-4`, `lr_warmup_steps: 4`, `lr_min_ratio: 1e-2`
- `model.model_config.scorer_scene_num: 1680`
- `trainer.max_epochs: 64`, `check_val_every_n_epoch: 16`
- `trainer.precision: bf16-mixed`, `gradient_clip_val: 1.0`,
  `accumulate_grad_batches: 1`
- `data.val_batch_size: 12`, `data.test_batch_size: 12`, `num_workers: 4`
- `data.train_memory_balanced_batching: true`,
  `trainer.use_distributed_sampler: false`

SMART NTP decoder는 static map feature를 시간 step마다 복제하지 않고, 모든 token step의
agent node가 같은 map feature를 참조한다. 지도 자체는 시간에 따라 바뀌지 않고, traffic
light처럼 시간 의존적인 값은 map-agent edge feature로 들어가므로, 이 방식은 objective를
바꾸지 않으면서 A100x4x2 학습의 map-agent attention 메모리 사용량을 줄인다.
이때 agent node는 시간축으로 펼쳐져 scene id 순서가 step마다 반복되므로, map-agent
radius graph를 만들기 전에 batch id를 정렬하고 edge index를 원래 순서로 되돌린다.
이 보정이 없으면 다른 scene의 지도가 섞이지는 않더라도, 같은 scene 안에서 agent가
봐야 할 map edge가 일부 조용히 누락될 수 있다. 따라서 현재 main의 SMART NTP 학습과
추론은 static map token을 유지하면서도 각 agent/time-step이 의도한 같은-scene map
context를 빠짐없이 받도록 처리한다.

A100/H100x4x2 SMART NTP launcher는 기본으로 `CATK_ATTENTION_GRAPH_FP32=1`을 설정한다.
전역 precision은 그대로 `bf16-mixed`로 두지만, `AttentionLayer` 안의 PyG graph attention
aggregation만 fp32로 계산한다. 즉 q/k/v projection과 FFN 같은 dense 연산은 mixed precision
이득을 유지하고, edge-wise gather, relation key/value projection, sparse softmax, dropout,
message 생성, scatter aggregation으로 이어지는 graph attention 경로만 fp32로 올린다.
모델 구조, 파라미터 수, edge set, radius, loss target은 바뀌지 않는다. 바뀌는 것은
attention 내부 계산 dtype 경계뿐이다.

또한 `AttentionLayer`의 relation `LayerNorm -> key/value projection` 경로는 기본으로
`torch.compile` 친화 함수로 묶어 실행한다. 이 경로는
`CATK_COMPILE_ATTENTION_RELATION_KV=0`으로 끌 수 있다. 수식과 파라미터 수는 그대로이며,
relation key/value projection을 PyG `message()` 내부 작은 op로 남겨두지 않고 한 번에
계산하도록 정리한 최적화다. A100에서는 이 최적화가 step time을 줄였지만 peak memory를
늘리므로, 현재 `testa/testaa` 기본 batch는 OOM 여유를 고려해 12로 둔다.

이 preset은 `data.train_memory_balanced_batching=true`도 켠다. 이 sampler는 각 training
pickle의 agent 수, valid agent-step 수, map point 수를 metadata cache로 한 번 기록한 뒤,
agent가 많은 scene이 같은 rank-local batch에 몰리지 않도록 batch 순서만 다시 짠다.
학습 objective, 모델 구조, per-GPU `train_batch_size=12`, 전체 effective batch 96은
그대로 유지된다. 대신 random shuffle 순서가 바뀌므로 기존 run을 resume하더라도 bitwise로
완전히 같은 sample 순서는 아니다.

안전장치로, `data.train_memory_balanced_batching=false`를 주더라도 DDP 학습에서는
datamodule이 train dataloader에 `DistributedSampler`를 직접 넣는다. 따라서
`trainer.use_distributed_sampler=false` 상태에서 memory-balanced sampler를 끄더라도
8개 rank가 같은 training cache 전체를 반복해서 보는 상황은 발생하지 않는다. 이 fallback은
memory balancing만 끄고, rank별 data sharding은 유지한다.

metadata cache 기본 위치는 training cache 안의
`.catk_memory_balanced_metadata_v1.pkl`이다. 파일이 없으면 같은 파일 시스템에서 한
process만 cache를 만들고 나머지 rank는 기다린다. 이후 실행에서는 이 cache를 바로 읽으므로
학습 step 속도에는 추가 비용이 없다. 이 숨김 cache 파일은 dataset sample 목록에서
제외되도록 처리되어 다음 실행의 학습 데이터에 섞이지 않는다. A100x4x2 preset은 첫 생성 속도를 위해
`data.train_memory_balance_metadata_num_workers=8`을 사용한다. cache를 다른 곳에 두고
싶으면 `data.train_memory_balance_metadata_path=/path/to/metadata.pkl`로 지정한다. metadata
build 중 pod가 죽어 `.lock` 파일만 남은 경우에는 lock heartbeat가 끊긴 것으로 보고 기본
30초 뒤 자동 회수한다. 살아 있는 builder를 기다리는 rank도 1초 단위로 cache 생성을 다시
확인하므로, stale lock 때문에 학습 준비가 장시간 멈추는 경로를 피한다.

`de4f438` 이후의 map surface category 분리를 실제로 학습에 쓰려면 cache도 새 형식이어야
한다. 학습 시작 전에 training cache 몇 개를 읽어 `pt_token.type`에 `10`/`11` 또는
`pt_token.pl_type`에 `4`/`5`가 나오는지 확인한다. 값이 계속 `0~9`, `0~3` 범위에만 머물면
기존 cache일 가능성이 높고, 이 경우 코드는 실행되지만 speed bump와 driveway를 crosswalk와
분리해 배우는 효과는 얻을 수 없다.

이 launcher와 내부 실행 스크립트는 `pre_bc_a100x4x2` fit 실행에서
`trainer.accumulate_grad_batches=1`을 강제하고, `data.train_batch_size`는 A100 80GB에서
검증할 수 있는 범위인 `24` 이하의 양의 정수만 허용한다. `--accumulate-grad-batches 2`처럼
gradient accumulation을 켜는 override는 학습을 시작하기 전에 실패한다.

relation K/V compile 최적화 적용 후, `testa/testaa`, A100 4장 x 2 pod, validation off,
`CATK_ATTENTION_GRAPH_FP32=1`, `data.train_use_eval_agent_selection=true` 조건에서
batch capacity를 다시 측정했다.
training cache `486,996`개 기준으로 64 epoch train-only 시간을 추정하면 아래와 같다.

| per-GPU batch | 결과 | profiler 기준 step time | 속도 | 관측 peak memory | 64 epoch train-only 예상 |
|---:|---|---:|---:|---:|---:|
| 13 | CUDA OOM | 첫 train step 부근 실패 | - | 약 `78,811 MiB` 후 추가 `784 MiB` 할당 실패 | 사용 안 함 |
| 12 | 통과 | `654.6 ms/step` | `1.53 it/s` | 약 `75,181 MiB` | 약 `59.0시간` |

relation K/V compile 경로는 step 속도를 줄이지만 peak memory도 늘린다. A100 80GB 장기
pretrain에서는 batch 13이 OOM이므로, 기본값은 안정성을 우선해 batch 12로 둔다. 같은
profiling 조건에서 기존 relation K/V 경로의 batch 13은 `768.4 ms/step`, 64 epoch train-only
약 `64.0시간`으로 측정되었고, compile 경로 batch 12는 batch가 낮아져 step 수가 늘어도
전체 train-only 예상 시간이 약 `59.0시간`으로 줄었다.

따라서 현재 A100x4x2 `testa/testaa` pretrain 기본값은
`CATK_ATTENTION_GRAPH_FP32=1`, `data.train_use_eval_agent_selection=true`,
`CATK_COMPILE_ATTENTION_RELATION_KV=1`, `data.train_batch_size=12`이다. 위 시간은
train-only 추정이며,
`check_val_every_n_epoch=16` validation과 checkpoint overhead는 별도로 더해진다.
RMM checkpoint 선택용 fast scorer는 `model.model_config.scorer_scene_num=1680`을 기준으로
validation batch 수를 자동 계산한다. A100x4x2 기본 `val_batch_size=12`, world size 8에서는
rank당 18 batch, 전체 약 1680개 scenario가 RMM 계산에 들어간다. 64 epoch 학습에서는
validation이 16 epoch마다 실행되어 총 4번의 checkpoint 후보를 만든다.

기본 cache root는 pod별로 다르다.

- `testa`: `/workspace/womd_v1_3/SMART_cache`
- `testaa`: `/workspace/womd_v1_3/SMART_cache`

다른 위치를 쓰려면 아래처럼 pod별로 override한다.

```bash
python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --task-name smart_ntp_pretrain_a100x4x2_fair \
  --pod-cache-root testa=/path/to/SMART_cache \
  --pod-cache-root testaa=/path/to/SMART_cache
```

시작 후 tmux에 붙으려면 launcher가 출력하는 attach 명령을 사용한다.

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-smart-ntp-a100x4x2
kubectl exec -it -n p-pnc testaa -c main -- tmux attach -t catk-smart-ntp-a100x4x2
```

실행을 멈추려면 같은 task name으로 stop을 호출한다.

```bash
python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --stop \
  --task-name smart_ntp_pretrain_a100x4x2_fair
```

### SMART pretrain 중단 후 resume

SMART pretrain은 두 종류의 checkpoint를 저장한다. Lightning `ModelCheckpoint`는
validation metric 기준 best checkpoint와 `last.ckpt`를 저장하고, 별도
`EpochLastCheckpointCallback`은 매 epoch의 학습 상태를
`checkpoints/epoch_last.ckpt` 하나로 계속 갱신한다. `epoch_last.ckpt`는 모델
weight뿐 아니라 optimizer, scheduler, epoch, global step, callback state를 포함하므로
중단된 pretrain을 이어갈 때 사용하는 기본 checkpoint이다.

가장 명시적인 resume 방법은 중단된 run의 `epoch_last.ckpt`를 직접 넘기는 것이다.

```bash
python -m src.run \
  experiment=pre_bc \
  action=fit \
  task_name=smart_ntp_pretrain_fair \
  ckpt_path=/path/to/logs/smart_ntp_pretrain_fair/runs/2026-05-16_12-00-00/checkpoints/epoch_last.ckpt
```

`ckpt_path`를 직접 지정하면 이 값이 항상 우선한다. A100x4x2 launcher에서도 같은
방식으로 넘길 수 있다.

```bash
python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --task-name smart_ntp_pretrain_a100x4x2_fair \
  --ckpt-path /mnt/nuplan/projects/catk/logs/smart_ntp_pretrain_a100x4x2_fair/runs/2026-05-16_12-00-00/checkpoints/epoch_last.ckpt
```

같은 task name 아래에서 가장 최신 `epoch_last.ckpt`를 자동으로 찾아 이어가려면
`resume.auto=true`를 켠다. 이때 코드는
`logs/<task_name>/runs/*/checkpoints/epoch_last.ckpt` 중 수정 시간이 가장 최신인
파일을 찾아 `trainer.fit(..., ckpt_path=...)`에 넘긴다.

```bash
python -m src.run \
  experiment=pre_bc \
  action=fit \
  task_name=smart_ntp_pretrain_fair \
  resume.auto=true
```

A100x4x2 launcher에서는 아래처럼 쓴다.

```bash
python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --task-name smart_ntp_pretrain_a100x4x2_fair \
  --auto-resume
```

다른 task 이름의 checkpoint를 찾아 이어가려면 `resume.task_name=<old_task_name>` 또는
launcher의 `--resume-task-name <old_task_name>`을 사용한다. 자동 resume은 기본적으로
checkpoint가 없으면 에러를 내고 새 학습을 시작하지 않는다. checkpoint가 없을 때 새로
시작하는 동작을 원하면 `resume.require_checkpoint=false` 또는
`--allow-missing-resume-checkpoint`를 명시한다.

중요한 제약은 resume 시 모델/config가 checkpoint와 호환되어야 한다는 점이다. 예를 들어
SMART 공정 비교 pretrain은 `num_freq_bands: 66`을 쓰므로, resume도 같은
`experiment=pre_bc` 또는 `pre_bc_a100x4x2` 계열 설정으로 실행해야 한다.

학습 batch, learning rate, epoch 수 같은 실험 파라미터를 바꿔야 할 때는 launcher option을
명시할 수 있다. 단, `pre_bc_a100x4x2` fit preset은 공정 비교용 batch 조건을 보호하기 위해
`trainer.accumulate_grad_batches=1`을 유지한다. `data.train_batch_size`는 기본 15이고,
OOM fallback 확인을 위해 24 이하의 값만 허용한다. 논문용 공정 비교가 아닌 별도 ablation에서
이 범위를 벗어나려면 다른 experiment preset을 따로 만들고, 어떤 값을 바꿨는지 KFM 쪽
recipe와 함께 기록해야 한다.

```bash
python scripts/launch_smart_ntp_a100x4x2_testa.py \
  --replace \
  --experiment pre_bc \
  --task-name smart_ntp_pretrain_a100x4x2_custom_lr \
  --learning-rate 5e-4
```

실제로 pod에 명령을 보내기 전에 무엇이 실행될지만 확인하려면 `--dry-run`을 붙인다.

공정 비교에서는 어느 epoch의 가중치를 비교하는지도 맞춰야 한다.
`configs/experiment/pre_bc.yaml`은 KFM pretrain 설정과 같이
`val_closed/sim_agents_2025/realism_meta_metric`을 checkpoint monitor로 사용하고
`mode: max`를 적용한다. 따라서 SMART NTP pretrain도 마지막 epoch 가중치가 아니라
closed-loop validation realism 점수가 가장 높았던 가중치를 best checkpoint로
저장한다.

공정 비교용 SMART NTP pretrain에서는 validation video 저장도 끈다.
`configs/experiment/pre_bc.yaml`은 `n_vis_batch: 0`, `n_vis_scenario: 0`,
`n_vis_rollout: 0`을 명시해서 validation 중 rollout video 저장이라는 불필요한
side-effect를 만들지 않는다.

같은 config는 SMART NTP의 capacity 보정을 위해
`model.model_config.decoder.num_freq_bands: 66`도 명시한다. 최신 main 코드에서
`experiment=pre_bc`와 `num_freq_bands=66`으로 SMART 모델을 실제 instantiate해 센
총 파라미터 수는 7,043,232개이며, 모두 trainable parameter이다.
순수 `configs/model/smart.yaml` 기본값인 `num_freq_bands=64` 기준 총 파라미터 수는
7,035,008개이다.

`local_val`, `wosac_sub`, `clsft`, `road_clsft`도 같은 pretrain checkpoint를 그대로
읽어야 하므로 `model.model_config.decoder.num_freq_bands: 66`을 명시한다. 이 값을
빠뜨리면 `pre_bc`에서 저장한 checkpoint의 Fourier embedding weight shape이 기본
SMART 값인 `64`와 맞지 않아 checkpoint load 단계에서 실패한다.

RMM 계산 scene 수는 `model.model_config.scorer_scene_num`을 기준으로 맞춘다. 기본 SMART와
공정 비교용 pretrain은 `1680`으로 설정되어 있으며, `configs/experiment/local_val.yaml`도
같은 값을 명시한다. `n_batch_sim_agents_metric`만 바꾸면 world size와 validation batch size에
따라 실제 scene 수가 달라질 수 있으므로, 논문용 RMM 비교에서 scene 수를 지정할 때는
`scorer_scene_num=<원하는 scene 수>`를 함께 설정한다. `wosac_sub`는 제출 파일 생성이
목적이므로 `scorer_scene_num: 0`으로 fast local metric을 끈다.

CAT-K와 RoaD fine-tuning config도 `n_vis_batch: 0`, `n_vis_scenario: 0`,
`n_vis_rollout: 0`을 명시한다. 공정 비교 실행에서는 validation video 저장이
불필요하고, debug 실행처럼 `val_batch_size`나 `n_rollout_closed_val`을 작게 줄였을 때
시각화 루프가 batch/rollout 개수보다 더 많이 접근하는 일을 피하기 위해서다.

### 학습 손실 metric 상태 정리

SMART의 학습/검증 손실은 모두 같은 `CrossEntropy` torchmetrics 인스턴스
(`self.training_loss`)를 통과한다. `training_step`과 `validation_step` 양쪽에서
`forward`로 호출되기 때문에 metric 내부의 `loss_sum`/`count` 상태가 phase를
넘나들며 누적되는 구조였다. 현재 코드 경로에서 `compute()`를 직접 호출하는
지점은 없어 가시적인 잘못된 숫자는 없었지만, DDP에서 `dist_reduce_fx="sum"`로
all-reduce가 trigger되거나 누가 epoch-mean 집계를 추가하면 즉시 train과 val의
loss가 섞여 잘못된 평균이 잡힐 수 있다.

이를 막기 위해 `on_train_epoch_start`와 `on_validation_start`에서
`self.training_loss.reset()`을 한 번씩 호출한다. 매 phase 시작 시 metric
buffer를 0으로 초기화하므로 train 누적과 val 누적이 서로를 오염시키지 않는다.
`forward` 반환값은 항상 현재 batch에 대한 값이라 `self.log("train/loss", ...)`나
`val_open/loss`로 찍히는 스칼라는 변하지 않는다.

### finetune/road_finetune checkpoint 로딩 안정성

`src/run.py`의 `finetune`과 `road_finetune` action은 SMART pretrain checkpoint를
`torch.load(...)["state_dict"]`로 직접 읽는다. PyTorch 2.6부터는
`torch.load`의 기본값이 `weights_only=True`로 바뀌면서 Lightning이 저장한
풀 checkpoint dict (state_dict 외 hyperparameter, callback state 등 포함)는
바로 unpickle되지 않는다. 두 action 모두 `weights_only=False`를 명시적으로
넘겨서 향후 torch 버전 업그레이드 시 finetune 경로가 silently 깨지는 일을
막는다.

### 2025 Sim Agents 제출 실행 시 빠른 지표 비활성화

`configs/experiment/wosac_sub.yaml`은 제출 파일 생성 전용 설정이다. 이 모드에서는
2025 Sim Agents submission protobuf를 저장하는 것이 목적이고, validation 중 fast Sim Agents
metric을 따로 누적해 로깅할 필요가 없다. 따라서 해당 config는
`n_batch_sim_agents_metric: 0`과 `scorer_scene_num: 0`을 명시해서 local fast metric
계산을 끈다.

이 config는 `semi_control_stable`의 WOSAC 제출 메타데이터를 main SMART NTP 기준으로
맞춘다. 기본값은 `authors: [Seulbin Hwang, Kiyoung Om]`,
`affiliation: NaverLabs`, `method_link: "not available yet"`,
`account_name: "h.sb@naverlabs.com"`을 사용한다. 모델과 방법론은 main의 SMART NTP이므로
`method_name`과 `description`은 SMART NTP 기준으로 둔다. Trainer/data 실행 자원 설정은
main의 기존 `wosac_sub` 기본값을 유지한다.

SMART 모델도 submission 모드에서는 scorer scene 수 자동 조정을 적용하지 않는다.
이렇게 하면 제출 파일의 내용은 유지하면서, 로깅되지 않을 metric state를 만들기
위해 앞쪽 validation batch를 불필요하게 평가하는 일을 피할 수 있다.

### 2025 Sim Agents 제출 조각 처리

SMART NTP main 브랜치도 KFM의 `semi_control_stable`과 같은 개념의
`SimAgentsSubmission` exporter를 사용한다. DDP validation/test dataloader는 각 rank가
서로 다른 scenario shard를 이미 한 번씩만 처리하므로, 제출 파일 생성 단계에서는 rank
간 rollout tensor를 다시 모으지 않는다. 각 rank는 자기 rank-local batch를 바로
`ScenarioRollouts`로 변환해 `sim_agents_2025_submission/` 아래에
`submission-rankXX-YYYYY.binproto` shard로 저장한다.

epoch end에서는 모든 rank가 남은 shard를 flush한 뒤 barrier로 저장 완료를 맞춘다.
그 다음 rank 0만 `sim_agents_2025_submission.tar.gz`를 만들고, tar 내부 member 이름은
`submission.binproto-00000-of-000NN` 형식으로 통일한다. 이 구조는 rank별 shard 경계와
최종 archive member naming을 명확히 해서 2025 Sim Agents 제출 기준에서 SMART NTP와
KFM의 평가/제출 파이프라인을 맞춘다.

`action=test`도 기본 설정에서는 `sim_agents_submission.is_active=false`이므로 제출
shard를 만들지 않는다. 이 경우 test loop는 distribution metric만 계산하고
`SimAgentsSubmission` exporter를 호출하지 않는다. 제출 파일을 만들려면
`experiment=wosac_sub`처럼 `model.model_config.sim_agents_submission.is_active=true`인
설정을 사용해야 한다. 이 guard는 실수로 기본 test 설정을 실행했을 때 inactive
exporter가 `update()`/`save_sub_file()`에서 깨지는 일을 막기 위한 안전장치이다.

Waymo Sim Agents 제출 형식은 scenario마다 32개 parallel simulation을 요구한다.
따라서 `configs/experiment/wosac_sub.yaml`은
`model.model_config.n_rollout_closed_val: 32`를 명시한다. 제출 exporter가 켜진 상태에서
이 값이 Waymo 제출 규격과 다르면 모델 초기화 단계에서 바로 중단해, 16 rollout 같은
잘못된 submission archive가 만들어지지 않도록 한다.

### 2025 Sim Agents 제출 메타데이터 placeholder 차단

`configs/model/smart.yaml`의 `sim_agents_submission` 기본값은 placeholder 상태(
`affiliation: YOUR_AFFILIATION`, `description: YOUR_DESCRIPTION`,
`method_link: YOUR_METHOD_LINK`, `account_name: YOUR_ACCOUNT_NAME`, `authors: [Anonymous]`)이다.
Waymo 리더보드는 이 placeholder 값이 그대로 들어간 archive를 거절한다.

이를 막기 위해 `SimAgentsSubmission`은 `is_active=True`로 켜졌을 때 메타데이터에 default
placeholder가 그대로 남아 있는지 모델 초기화 단계에서 검사한다. 하나라도 남아 있으면
clear한 한국어 메시지와 함께 `ValueError`로 즉시 중단하므로, 몇 시간짜리 rollout이 끝난
뒤에야 잘못된 tar.gz가 생기는 일이 없다. `configs/experiment/wosac_sub.yaml`은
`semi_control_stable`에서 가져온 제출 계정/저자/기관 값을 기본으로 override하므로 그대로
실행할 수 있다. 다른 계정이나 다른 제출명으로 올릴 때만 README §"SSH 서버에서 Waymo
사이트로 자동 업로드" 예시처럼 `model.model_config.sim_agents_submission.*` 필드를
override하면 된다.

### Submission proto의 `num_model_parameters` 자동 설정

`SimAgentsChallengeSubmission` proto의 `num_model_parameters` 문자열은 모델 yaml에서
직접 지정한다. `configs/model/smart.yaml`은 SMART-tiny에 맞춰 `"7M"`, `smart_mini_3M.yaml`은
`"3M"`, `smart_nano_1M.yaml`은 `"1M"`을 기본값으로 설정한다. 다른 모델 크기를 쓸 때는
`model.model_config.sim_agents_submission.num_model_parameters="<크기>"`를 함께 override
하면 archive 메타데이터가 실제 모델 크기와 어긋나지 않는다. 이 값을 비워 두지는 못하며
submission proto에 그대로 기록된다.

### Submission scenario dedup은 set 기반

`SimAgentsSubmission.aggregate_rollouts`는 같은 scenario가 두 번 들어가지 않도록
`submission_scenario_id`로 중복을 거르는데, 이를 set으로 운영해서 검사 비용을 O(1)로
유지한다. 전체 WOMD test split 같은 큰 평가에서도 rank당 누적 scenario 수와 무관하게
aggregate 단계가 O(n)에 끝난다.

로컬 실행 스크립트는 `scripts/setup_runtime_env.sh`를 통해 conda 환경과 cache root를
찾는다. 기본으로 현재 머신의 `/media/user/E/dataset/womd_v1_3/SMART_cache`가 있으면
그 경로를 사용하고, 없으면 `/scratch/cache/SMART`를 사용한다. 다른 경로를 쓰려면
`CACHE_ROOT=/path/to/SMART_cache`를 지정하면 된다. `local_val.sh`와 `wosac_sub.sh`는
평가할 checkpoint가 필요하므로 `CKPT_PATH=/path/to/model.ckpt`를 함께 지정해야 한다.

### SSH 서버에서 Waymo 사이트로 자동 업로드

SSH 서버에서도 `sim_agents_2025_submission.tar.gz`를 만든 뒤 바로 Waymo 사이트에 업로드할 수
있다. Google 로그인은 한 번 필요하므로, GUI가 있는 PC에서 로그인 상태를 저장한 뒤
서버에서는 그 JSON 내용을 붙여넣는 방식으로 쓰는 편이 안전하다. 로그인 상태 파일의
기본 위치는 아래와 같다.

```text
secrets/waymo/waymo_storage_state.json
```

이 파일은 로그인된 상태를 담고 있으므로 비밀번호처럼 다뤄야 한다. `.gitignore`에는
`secrets/waymo/waymo_storage_state.json`과 `secrets/waymo/playwright_profile/`이
포함되어 있다.

준비:

```bash
python -m pip install -r install/requirements.txt
python -m playwright install chromium
```

환경에 `python` 명령이 없으면 위 예시의 `python`을 `python3`로 바꿔서 실행하면 된다.

GUI가 있는 PC에서 로그인 상태를 저장한다.

```bash
python scripts/waymo_save_storage_state.py --browser-channel chrome
```

기본 저장 위치는 `secrets/waymo/waymo_storage_state.json`이다. 로그인이 잘 안 되면
Playwright 기본 Chromium보다 설치된 Chrome이나 Edge를 쓰는 편이 안정적이므로
`--browser-channel chrome` 또는 `--browser-channel msedge`를 권장한다. 이 스크립트는
저장 직전에 Sim Agents 페이지에서 `Submit to Validation Set` 또는 `Submit to Test Set`
업로드 박스가 실제로 보이는지 확인한다. Waymo가 `Review rules`를 보여주면 약관 동의를
마친 뒤 다시 저장해야 한다. 이때 headless 업로드에 필요한 Waymo localStorage 항목인
`datasetChallengeTermsAgreementAccepted=true`도 storage state에 함께 기록한다.

브라우저 프로필은 기본적으로 실행할 때마다 임시로 만들고 종료 시 정리한다.
`--user-data-dir`를 직접 줄 때는 Playwright 전용의 빈 폴더를 쓰는 편이 안전하며,
평소 쓰는 기본 Chrome 프로필 폴더를 그대로 넣는 것은 권장하지 않는다. 예전에 만든
프로필을 재사용하다가 브라우저가 바로 꺼지면 `--user-data-dir` 없이 다시 실행하면 된다.

서버에 이 파일을 꼭 복사해 둘 필요는 없다. `waymo_submission.enabled=true`로 실행했을
때 파일이 없으면 rank 0 프로세스가 시작 직후 JSON 붙여넣기를 요청한다. pretty-printed
JSON을 그대로 붙여넣고 마지막 `}` 뒤에서 Enter를 한 번 더 치면 된다. 입력된 JSON은
`/tmp` 아래 임시 파일로만 저장되고 프로세스 종료 시 삭제된다. 서버에 파일을 두고 싶으면
`waymo_submission.storage_state_path` 경로에 배치하면 되고, 이 경우 붙여넣기 프롬프트는
뜨지 않는다.

validation 자동 업로드 예시는 아래와 같다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=wosac_sub \
  action=validate \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=smart_ntp_waymo_val_ddp6 \
  waymo_submission.enabled=true \
  waymo_submission.poll_submission_status=false
```

핵심 옵션:

- `waymo_submission.enabled=true`: 자동 업로드를 켠다.
- `waymo_submission.storage_state_path`: 로그인 상태 파일 경로이다. 기본값은
  `secrets/waymo/waymo_storage_state.json`이다.
- `waymo_submission.poll_submission_status=false`: 업로드 후 점수 페이지를 계속 확인하지
  않는다.

추가 참고:

- validation 실행에서는 `waymo_submission.enabled=true`만 주면 업로드까지 진행된다.
- `torchrun` DDP에서도 rank 0만 한 번 입력을 받고, 나머지 rank는 그 입력이 끝날 때까지
  대기한다.
- 서버에서는 기본으로 headless Chromium을 사용한다.
- 서버에 설치된 Chrome을 쓰고 싶으면 `waymo_submission.browser_channel=chrome` 또는
  `waymo_submission.browser_executable_path=/path/to/chrome`를 지정한다.
- Chromium launch 전에 `CONDA_PREFIX/lib`를 `LD_LIBRARY_PATH` 앞에 자동으로 추가하고,
  Playwright bundled browser 외에도 system Chrome과
  `~/.cache/ms-playwright/chromium-*/chrome-linux/chrome` 경로를 자동 탐색해 재시도한다.
- 브라우저 launch에 실패하면 저장된 storage state 쿠키를 사용해 Waymo 업로드 API로
  자동 fallback한다.
- 로그인 만료나 페이지 구조 변경으로 실패하면
  `logs/<task_name>/runs/<timestamp>/waymo_submission_debug/` 아래에 디버그 파일이 남는다.
- 점수 페이지까지 자동 확인하고 싶으면 `waymo_submission.poll_submission_status=true`를 줄
  수 있지만 UI 변경에 영향을 받을 수 있어 기본값은 `false`이다.

test 자동 제출은 실수 방지를 위해 기본으로 꺼져 있다. Waymo test set은 계정당 30일에
3번만 제출할 수 있으므로 test 업로드를 할 때는 아래 옵션을 추가로 넣어야 한다.

```bash
... action=test \
    waymo_submission.enabled=true \
    waymo_submission.submit_test=true
```

#### Waymo 자동 업로드 로그 메시지 안정화

`src/utils/waymo_submission.py`의 모든 `log.info`/`log.warning` 호출은 `%`-스타일
포맷 인자(`log.info("...%s...", value)`) 대신 f-string 으로 작성되어 있다. 이는
`RankedLogger.log(self, level, msg, rank=None, *args, **kwargs)` 시그니처가
첫 번째 추가 positional 인자를 `rank` 키워드로 흡수해 포맷 인자가 한 칸 밀려
경고 메시지가 깨지는 현상을 피하기 위함이다. 이 파일에 새 로그 호출을 추가할 때도
같은 패턴을 따라야 한다.

### SMART 기준 모델의 신호등 입력

`f6e96cf8`의 동적 traffic-light staleness 경로는 적용하지 않는다. 현재 SMART token
baseline은 traffic-light type을 map token의 static categorical feature로만 사용한다.
`SMARTMapDecoder`는 cache의 `light_type`을 map point embedding에 더하지만, agent-to-lane
relation에는 traffic-light state나 관측 경과 시간을 넣지 않는다.

따라서 map-to-agent relation은 `distance / bearing / relative heading` 3D 기하 정보만
사용한다. `prediction_time - observed_light_time` 같은 rollout-time-dependent staleness
scalar를 만들지 않으며, `NO_LANE_STATE`와 관측된 `UNKNOWN`도 별도 dynamic relation bias로
처리하지 않는다. Cache에 저장된 `light_type`은 static map-token feature로만 소비된다.

SMART closed-loop validation/test는 모든 `n_rollout_closed_val` rollout을 하나의
rollout-major batch로 실행한다. Map encoder는 한 번만 평가하고, 그 뒤 `pt_token`,
`position`, `orientation`, `batch`를 rollout-specific batch offset과 함께 확장한다.
Agent tensor도 같은 rollout 순서로 확장하며, output은 metric, WOSAC submission,
visualization code가 보기 전에 `[agent, rollout, time, ...]` 형태로 되돌린다. 이렇게 하면
공개 validation/test 결과 interface는 유지하면서 rollout마다 inference를 반복 호출하는 일을
줄일 수 있다.

CUDA 메모리가 부족해서 전체 rollout 묶음을 한 번에 실행하지 못하면, validation/test
코드는 rollout chunk 크기를 자동으로 줄여 다시 시도한다. 예를 들어
`n_rollout_closed_val=16`이면 먼저 16개를 한 번에 실행하고, CUDA OOM이 발생한
경우에만 `8 -> 4 -> 2 -> 1` 순서로 더 작은 묶음을 시도한다. 각 rollout seed는
`scenario_id`와 rollout index로 고정되므로 chunk 크기가 바뀌어도 같은 rollout
index는 같은 sampling seed를 사용한다. 이 기능은 속도 향상보다는 validation이
메모리 부족으로 중단되지 않게 하는 안정성 장치이다.

Stochastic SMART validation/test sampling은 `validation_closed_seed`, `scenario_id`,
rollout index에서 만든 rollout/scenario별 seed를 사용한다. 각 expanded scenario는
closed-loop token rollout 동안 자기 `torch.Generator`를 유지하므로, `topk_prob`
sampling은 rollout을 하나씩 실행하는지 큰 rollout batch로 실행하는지에 영향을 받지 않는다.

학습/검증 입력을 만드는 `TokenProcessor`의 map/agent token matching은 stochastic
top-k sampling을 쓰지 않는다. 지도 token은 항상 가장 가까운 token을 고르고, agent
token의 `sampled_*` 상태도 기존 기본값 `num_k=1, temp=1.0`과 동일하게
teacher-forced nearest-token `gt_*` 상태와 같게 만든다. 따라서 tokenization에는
별도 `map_token_sampling` / `agent_token_sampling` config가 없다. 위
validation/test sampling 설명은 closed-loop rollout 중 모델 출력 token을 고르는
정책에만 해당한다.

Agent tokenization에서 첫 valid step이 coarse token boundary와 어긋난 경우, 직전
token boundary까지 위치/heading/velocity/valid를 외삽해 최소 하나의 history token이
유효하도록 만든다. 이 외삽은 agent별 Python loop가 아니라 batch mask/index 연산으로
처리한다. 규칙은 기존 loop와 같고, `t=10`인데 raw step 5가 invalid인 history 보강
예외도 유지한다.

Agent token matching은 기존과 같은 teacher-forced recurrence를 유지한다. 다만 각
coarse step마다 모든 candidate token을 agent별 global 좌표로 펼쳐 비교하지 않고,
정답 contour를 직전 token 상태 기준 local 좌표로 한 번 옮긴 뒤 static local token bank와
직접 비교한다. rigid transform은 거리 순서를 보존하므로 선택되는 nearest token,
`gt_*`, `sampled_*` 의미는 기존 구현과 같다. 큰 batch에서 peak tensor가 튀지 않도록
query agent 축만 chunk 단위로 나눠 처리한다.

DDP validation/test에서는 각 rank가 validation/test sample을 복제 없이 정확히 한
번씩 나눠 처리한다. 일반 distributed sampler처럼 dataset 길이를 world size에 맞추기
위해 뒤쪽 sample을 padding 복제하지 않으므로, 멀티 GPU 평가에서 같은 scenario가
중복 채점되는 일을 막고 불필요한 validation work를 줄인다. 학습 dataloader의
shuffle/sampling 정책은 이 변경의 영향을 받지 않는다.

학습 중 checkpoint 확인만 빠르게 하고 싶으면 아래 옵션을 명시적으로 켤 수 있다.

```bash
model.model_config.fit_time_fast_validation_only=true \
model.model_config.val_open_loop=false
```

이 모드는 `fit` 실행 중 validation batch 수를 Fast WOSAC scorer에 필요한
`n_batch_sim_agents_metric` 값으로 제한한다. 즉, full validation과 같은 결과가
아니라 빠른 checkpoint 선택용 근사 validation이다. 그래서 open-loop validation과
submission 저장 모드에서는 자동으로 켜지지 않는다. 논문용 최종 RMM 비교, 전체
validation, WOSAC submission 생성은 이 옵션 없이 `validate` 또는 `test`로 다시
실행해야 한다.

활성 조건은 `val_closed_loop=true`, `val_open_loop=false`, submission inactive,
`n_batch_sim_agents_metric > 0`이다. `fit_time_fast_validation_only=true`만 켜고
이 중 하나라도 충족하지 않으면 fast 모드가 silently OFF가 되곤 했는데, 이제
모델 초기화 단계에서 `[fit_time_fast_validation_only] 옵션이 켜져 있지만 활성
조건이 충족되지 않아 fast 모드가 적용되지 않았습니다. 원인: ...` 메시지를 한 번
출력해 어떤 조건이 빠졌는지 바로 알 수 있도록 한다.

### SMART 검증 로거와 영상 정리

SMART validation은 trainer에 등록된 logger 중 `log_video`를 지원하는 logger를 찾아
visualization 영상을 기록한다. logger가 없거나 현재 기본 logger가 video logging을
지원하지 않아도 validation metric 계산 자체가 실패하지 않도록, video logging은 가능한
경우에만 수행한다. epoch metric logging도 logger가 있을 때만 호출한다.

기본 설정은 `model.model_config.delete_local_videos_after_wandb_upload=true`이다.
따라서 video logger에 mp4를 넘긴 뒤에는 로컬 mp4 파일과 비어 있는 video 하위
디렉터리를 정리한다. 반복 validation에서 `logs/.../videos` 아래 파일이 계속 쌓이는
것을 막기 위한 운영 옵션이며, rollout 결과나 metric 값은 바꾸지 않는다. 로컬 mp4를
남겨서 직접 확인하고 싶으면 이 값을 `false`로 덮어쓰면 된다.

### SMART 기준 모델의 움직임 결측 특징

SMART next-token baseline은 이제 kinematic-control flow 실험과 같은 방식으로 agent
context에 motion missingness를 드러낸다. 각 agent motion feature는
`[coarse displacement norm, heading-relative displacement angle, motion_valid]`이다.
유효하지 않은 coarse displacement는 0으로 채우고, `motion_valid=0`은 이 0 displacement가
실제 정지가 아니라 motion missing에서 온 값임을 모델에 알려준다.

Agent-to-agent relation feature는 edge마다 다시 motion을 붙이지 않고 기존 기하 정보
`distance / bearing / relative heading`만 사용한다. 대신 a2a radius graph를 만들 때
유효하지 않은 agent state는 애초에 neighbor 후보에서 제외한다. 따라서 "실제 정지"와
"missing이라서 0으로 채운 motion"의 구분은 agent node feature에만 존재하고, 주변 agent의
missingness 정보는 attention의 sender/receiver hidden state를 통해 전달된다. 이 구조는
08a2e89의 missingness 의미는 유지하면서 edge 수에 비례하던 relative-motion embedding
비용을 제거한다.

이 변경은 SMART token validity나 loss masking을 바꾸지 않는다. SMART는 여전히 두 coarse
endpoint token이 valid일 때 0.5초 token을 valid로 본다. Control-flow branch는 더 엄격한
fine-step segment validity를 사용할 수 있다. 이 변경은 shared context feature에서 사용할 수
있는 missingness 정보만 맞춘다. Agent node embedding input dimension은 motion_valid 추가로
바뀌었기 때문에, 이 변경 이전의 SMART checkpoint는 새 pretraining run과 shape이 호환되지
않는다. 또한 08a2e89처럼 6D a2a relation을 사용한 중간 checkpoint도 현재 3D a2a relation
구조와 shape이 맞지 않는다.

### WOSAC-CPD / WOSAC-CES 분포 지표

Closed-loop validation과 WOSAC 제출 export는 이제 모델이 이미 생성한 10Hz rollout에서
분포 지표도 계산한다. 이 지표만을 위해 추가 rollout을 만들지는 않는다.

- `val_closed/WOSAC-CPD/value`: 같은 scenario에서 나온 rollout 사이의 conditional pairwise diversity이다. 값이 높을수록 rollout이 더 다양하다.
- `val_closed/WOSAC-CES/value`: conditional Energy Score 계열 distribution metric이다. Validation GT future가 있을 때만 계산된다. 낮을수록 좋다.
- `test/WOSAC-CPD/value`: test submission export 중 측정한 CPD이다. Test split에는 GT future가 없으므로 CES는 보고하지 않는다.
- `*/WOSAC-CPD/DPR`: diversity preservation ratio이다. `model.model_config.wosac_cpd_reference`에 양수 pretrain CPD 값을 넣었을 때만 logging된다.

두 metric은 같은 WOSAC joint rollout distance를 사용한다. Validation CPD/CES는 validation GT에서 자동 계산한 future-motion scale로 agent type별 normalize를 수행한다. 이 scale은 metric 내부에서 계산되며 사용자가 정하는 threshold가 아니다.

**주의**: `test/WOSAC-CPD/value`는 GT가 없어 agent type별 scale을 계산할 수 없으므로 정규화 없이 raw 단위로 보고된다. `val_closed/WOSAC-CPD/value`(0~1 근방의 정규화 값)와 절대값으로 비교하면 안 된다. 같은 split끼리의 상대 비교에만 사용한다.

CAT-K fine-tuning의 diversity preservation을 보고하려면 아래처럼 실행한다.

```bash
python -m src.run \
  experiment=clsft \
  ckpt_path=/path/to/catk.ckpt \
  model.model_config.wosac_cpd_reference=<SMART_PRETRAIN_CPD>
```

`n_rollout_closed_val=32`이면 metric은 32개 rollout을 사용한다. 이 값을 16으로 바꾸면 이미 생성된 16개 rollout을 사용한다.

### RoaD 미세조정

이 저장소는 아래 config를 통해 RoaD 스타일 closed-loop fine-tuning도 지원한다.

```text
configs/experiment/road_clsft.yaml
```

RoaD fine-tuning은 epoch-local generated dataset pipeline으로 구현되어 있다.

- 원본 WOMD training pickle cache는 절대 덮어쓰지 않는다.
- 각 fine-tuning epoch 시작 시 최신 SMART 모델이 원본 training cache에서 임시 RoaD cache를 생성한다.
- 기본적으로 scenario마다 독립적인 RoaD rollout 3개를 만든다.
- Training dataset은 각 scenario에서 생성된 3개 rollout 중 하나를 균등 sampling하므로, effective length는 원본 scenario 수와 같다.
- 임시 cache는 매 epoch 다시 생성되고 기본적으로 사용 후 삭제된다.
- 학습 종료 후 datamodule의 train cache 경로는 원본 WOMD training cache로 되돌린다.

기본 RoaD 설정은 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| config | `road_clsft` |
| 실행 action | `road_finetune` |
| 학습률 | `5e-5` |
| label smoothing | `0.0` |
| scenario당 rollout 수 | `3` |
| candidate 정책 | Sample-K |
| candidate 수 | `64` |
| sampling 온도 | `0.8` |
| candidate 선택 | 정답 다음 상태에 가장 가까운 sampled token |
| RoaD 학습 중 step 내부 CAT-K rollout | 꺼짐 |

명시적인 SMART BC checkpoint로 RoaD fine-tuning을 실행하려면 아래처럼 실행한다.

```bash
torchrun -m src.run \
  experiment=road_clsft \
  ckpt_path=/path/to/SMART_BC_PRETRAINED.ckpt \
  task_name=road_clsft
```

또는 아래 script를 사용할 수 있다.

```bash
bash scripts/road_train.sh
```

DDP 학습에는 `scripts/road_train.sh` 안의 DDP block을 사용한다. Rank 0이
`${paths.output_dir}/road_cache/epoch_XXX` 아래에 epoch-local RoaD cache를 생성하고,
그 뒤 모든 rank가 synchronize한 다음 해당 cache를 읽는다. 생성된 데이터를 debug하려는
경우에만 `road.delete_after_use=false`를 설정한다.

#### RoaD cache 정책: 후처리 방어 없이 그대로 학습 + precision 정합

RoaD는 모델 자기 자신의 rollout을 그대로 새 정답으로 다시 학습시키는 방식이다.
rollout이 도로 밖으로 폭주하거나 ego로부터 멀리 벗어나도, 캐시 단계에서는 거리 클립이나
도로 이탈 같은 후처리 방어 로직을 **일부러 두지 않는다**. 모델이 만든 80 step rollout
전체가 그대로 학습 신호가 된다. 우리는 항상 `data.train_use_eval_agent_selection=true`로
학습하므로, 학습 transform이 거리를 잘라주는 가정도 없다.

다만 `RoadCacheRefreshCallback`은 `trainer.precision`을 읽어 `bf16-mixed`/`16-mixed`일 때
`generate_road_cache` 안의 inference를 `torch.autocast`로 감싼다. 학습 step과 cache
생성이 같은 precision 분포로 돌아가도록 정합을 맞추기 위한 것이며, `32-true`로 실행할
때는 autocast가 깔리지 않는다. 이는 후처리 방어 로직이 아니라 precision 정합 목적이다.

Gaussian Mixture Model(GMM) 기반 ego policy도 절차는 비슷하며, 아래 config를 사용하면 된다.

- [GMM 기반 ego policy용 BC pre-training config](configs/experiment/ego_gmm_pre_bc.yaml)
- [GMM 기반 ego policy용 CLSFT with CAT-K config](configs/experiment/ego_gmm_clsft.yaml)
- [GMM 기반 ego policy용 local validation config](configs/experiment/ego_gmm_local_val.yaml)
- Ego-policy에는 submission 옵션이 없다.

## 성능

CAT-K로 fine-tune한 SMART의 [WOSAC 리더보드](https://waymo.com/open/challenges/2024/sim-agents/) submission은 [여기](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731338655639000/)에서 확인할 수 있다.
재현한 SMART의 test split submission은 [여기](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731391949275000/)에서 확인할 수 있다. 이 submission은 리더보드에 공개하지 않았다.

## 절제 실험 설정

Ablation model 설정은 [docs/ablation_models.md](docs/ablation_models.md)를 참고하면 된다.
특히 [SMART](https://arxiv.org/abs/2207.05844)와 [Trajeglish](https://arxiv.org/abs/2312.04535)가 사용한 data augmentation 방법을 확인할 수 있다.

## 감사의 말

이 코드는 [SMART](https://github.com/rainmaker22/SMART)를 기반으로 한다. 가치 있는 open-source code를 공개해 준 SMART 저자들에게 감사드린다. 이 훌륭한 연구도 함께 인용해 주면 좋다.
