# Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models


<p align="center">
     <img src="docs/catk_banner.png" alt="Closest Among Top-K (CAT-K) rollouts unroll the policy during fine-tuning in a way that visited states remain close to the ground-truth.", width=760px>
     <br/><strong>Closest Among Top-K (CAT-K) Rollouts</strong> unroll the policy during fine-tuning in a way that visited states remain close to the ground-truth (GT). At each time step, CAT-K first takes the top-K most likely action tokens according to the policy, then chooses the one leading to the state closest to the GT. As a result, CAT-K rollouts follow the mode of the GT (e.g., turning left), while random or top-K rollouts can lead to large deviations (e.g., going straight or right). Since the policy is essentially trained to minimize the distance between the rollout states and the GT states, the GT-based supervision remains effective for CAT-K rollouts, but not for random or top-K rollouts.
</p>

> **Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models**            
> [Zhejun Zhang](https://zhejz.github.io/), [Peter Karkus](https://karkus.tilda.ws/), [Maximilian Igl](https://maximilianigl.com/), [Wenhao Ding](https://wenhao.pub/), [Yuxiao Chen](https://research.nvidia.com/labs/avg/author/yuxiao-chen/), [Boris Ivanovic](https://www.borisivanovic.com/) and [Marco Pavone](https://web.stanford.edu/~pavone/index.html).<br/>
> 
> [Project Page](https://zhejz.github.io/catk)<br/>
> [arXiv Paper](https://arxiv.org/abs/2412.05334)

```bibtex
@inproceedings{zhang2025closed,
  title = {Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models},
  author = {Zhang, Zhejun and Karkus, Peter and Igl, Maximilian and Ding, Wenhao and Chen, Yuxiao and Ivanovic, Boris and Pavone, Marco},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
}
```

## News & Updates

Apr. 2025
- **Oral at CVPR 2025**: Cheers!
- **Waymo Sim Agents 2025 ready**: This branch evaluates closed-loop rollouts with the Waymo Sim Agents 2025 metric path only.

Feb. 2025
- **Paper accepted at CVPR 2025:** Cheers!

- **Model checkpoints for Sim Agents:** You can obtain the checkpoints for our Sim Agents submission (SMART-tiny-CLSFT) by sending an email to Zhejun (zhejun.zhang94@gmail.com). In accordance with Waymo's terms, you must attach a screenshot showing that you are registered and logged into the [My Submissions](https://waymo.com/open/challenges/submissions) page of the Waymo Open Dataset.

- **SMART-mini and SMART-nano:** SMART-tiny with 7M parameters requires training on 8x A100 for a few days, which may be unaffordable in some cases. To address this, we have added config files for two smaller model, [smart_mini_3M.yaml](configs/model/smart_mini_3M.yaml) and [smart_nano_1M.yaml](configs/model/smart_nano_1M.yaml). Specifically, SMART-nano-1M can be trained on a single A100, but its performance is significantly worse. After pre-training and CAT-K fine-tuning, we achieved an RMM of 0.74 with SMART-nano-1M, which is 0.03 lower than that of SMART-tiny-7M. 

Jan. 2025
- **Sim Agents benchmark note:** CAT-K resolves an issue in the agent token vocabulary and improves SMART-tiny closed-loop realism. This SMART branch keeps evaluation on the Waymo Sim Agents 2025 metric path.

- **Issue in the agent token vocabulary:** We discovered that the [agent token vocabulary file](src/smart/tokens/cluster_frame_5_2048_remove_duplicate.pkl) we were using (borrowed from the [SMART repository](https://github.com/rainmaker22/SMART/blob/main/smart/tokens/cluster_frame_5_2048.pkl)) was intended only for sanity checks and not for reproducing optimal performance. To resolve this, we added a [script](src/smart/tokens/traj_clustering.py) and used it to build an [appropriate agent token vocabulary](src/smart/tokens/agent_vocab_555_s2.pkl). Our script is based on the [k-disk clustering script from SMART](https://github.com/rainmaker22/SMART/blob/main/scripts/traj_clstering.py). Thanks to the updated agent tokens, all our traffic simulation models saw a significant performance improvement of approximately +0.0060 RMM!



## Installation
- The easy way to setup the environment is to create a [conda](https://docs.conda.io/en/latest/miniconda.html) environment using the following commands
  ```
  conda create -y -n catk python=3.11.9
  conda activate catk
  conda install -y -c conda-forge ffmpeg=4.3.2
  pip install -r install/requirements.txt
  pip install torch_geometric
  pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
  pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
  ```
- Alternatively, a better way is to use the [Dockerfile](install/Dockerfile) and build your own docker. We found the code runs faster in the docker for some reasons.
- We use [WandB](https://wandb.ai/) for logging. You can register an account for free.
- **Be aware**
  - We use 8 *NVIDIA A100 (80GB)* for training and validation, the training and fine-tuning take a few days, whereas the validation and testing take a few hours.
  - We cannot share pre-trained models according to the [terms](https://waymo.com/open/terms) of the Waymo Open Motion Dataset.


## Dataset preparation
- Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/). We use v1.2.1.
- Use [scripts/cache_womd.sh](scripts/cache_womd.sh) to preprocess the dataset into pickle files to accelerate data loading during the training and evaluation.
- You should pack three datasets: `training`, `validation` and `testing`.

### 10.60.188.83 SMART RAW cache 생성

`SMART` 브랜치 기준 WOMD raw scenario를 새 SMART cache로 만들 때는 아래 경로를 사용한다.

```text
raw input:    /media/user/E/dataset/womd_v1_3/scenario
cache output: /media/user/F/dataset/womd_v1_3/SMART_RAW_cache
log output:   /media/user/F/dataset/womd_v1_3/SMART_RAW_cache_build_logs
```

원격 머신에서 장기 작업으로 실행한다.

```bash
ssh user@10.60.188.83
tmux attach -t smart-raw-cache-build
```

세 split을 직접 다시 생성해야 하면 아래 명령을 같은 방식으로 실행한다. `validation` split은 validation metric과 submission에 필요한 `validation_tfrecords_splitted`도 함께 만든다.

```bash
source /media/user/E/miniforge/etc/profile.d/conda.sh
conda activate catk
cd /tmp/catk_smart_cache_build

RAW_ROOT=/media/user/E/dataset/womd_v1_3/scenario
CACHE_ROOT=/media/user/F/dataset/womd_v1_3/SMART_RAW_cache
WORKERS=112

python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split training --num_workers "$WORKERS"
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split validation --num_workers "$WORKERS"
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split testing --num_workers "$WORKERS"
```

완료 후 최소 검증은 아래를 확인한다.

```bash
find /media/user/F/dataset/womd_v1_3/SMART_RAW_cache/training -maxdepth 1 -name '*.pkl' | wc -l
find /media/user/F/dataset/womd_v1_3/SMART_RAW_cache/validation -maxdepth 1 -name '*.pkl' | wc -l
find /media/user/F/dataset/womd_v1_3/SMART_RAW_cache/testing -maxdepth 1 -name '*.pkl' | wc -l
find /media/user/F/dataset/womd_v1_3/SMART_RAW_cache/validation_tfrecords_splitted -maxdepth 1 -name '*.tfrecords' | wc -l
grep -R "Traceback\\|Exception\\|No space left" /media/user/F/dataset/womd_v1_3/SMART_RAW_cache_build_logs || true
```

### 10.60.188.83 SMART RAW cache Nubes 업로드

생성된 SMART RAW cache를 Nubes에 업로드할 때는 [scripts/upload_smart_raw_cache_to_nubes.sh](scripts/upload_smart_raw_cache_to_nubes.sh)를 사용한다.

기본 경로는 아래와 같다.

```text
local cache: /media/user/F/dataset/womd_v1_3/SMART_RAW_cache
nubes path:  labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_RAW_cache
jobs:        96
gateway:     c.nubes.sto.navercorp.com:8000
```

원격 머신에서 실행한다.

```bash
ssh user@10.60.188.83
cd /tmp/catk_smart_cache_build
bash scripts/upload_smart_raw_cache_to_nubes.sh
```

장기 작업으로 돌릴 때는 tmux를 사용한다.

```bash
tmux new -s smart-raw-cache-upload
bash scripts/upload_smart_raw_cache_to_nubes.sh
```

업로드가 중간에 끊긴 경우 같은 명령을 다시 실행하면 된다. 기본 옵션은 `-s`라서 Nubes에 이미 있는 파일은 건너뛴다. 병렬도는 필요할 때만 조정한다.

```bash
JOBS=96 bash scripts/upload_smart_raw_cache_to_nubes.sh
```

### SMART RAW cache Nubes 다운로드

Nubes에 업로드된 SMART RAW cache를 학습 파드로 내려받을 때는 [scripts/download_smart_raw_cache_from_nubes.sh](scripts/download_smart_raw_cache_from_nubes.sh)를 사용한다.

기본 경로는 아래와 같다.

```text
nubes path:  labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_RAW_cache
pod cache:   /workspace/womd_v1_3/SMART_RAW_cache
jobs:        96
gateway:     c.nubes.sto.navercorp.com:8000
```

파드 안에서 직접 실행하는 기본 명령은 아래와 같다.

```bash
NUBES_JOBS=96 bash scripts/download_smart_raw_cache_from_nubes.sh
```

이미 원격 cache가 정상 업로드된 것을 확인했고, 가장 빠르게 내려받는 것이 목적이면 원격 파일 목록 생성 단계를 생략한다.

```bash
SKIP_REMOTE_LIST=1 NUBES_JOBS=96 bash scripts/download_smart_raw_cache_from_nubes.sh
```

`testa`, `testaa`, `testas` 같은 Kubernetes 파드에 동일 cache를 내려받을 때는 스크립트를 파드에 복사한 뒤 tmux로 장기 실행한다. 예시는 `testa` 기준이다.

```bash
kubectl cp -n p-pnc scripts/download_smart_raw_cache_from_nubes.sh \
  testa:/tmp/download_smart_raw_cache_from_nubes.sh \
  -c main

kubectl exec -n p-pnc testa -c main -- bash -lc '
chmod +x /tmp/download_smart_raw_cache_from_nubes.sh
tmux new-session -d -s smart-raw-cache-download "
  SKIP_REMOTE_LIST=1 \
  NUBES_JOBS=96 \
  REMOTE_DIR=labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_RAW_cache \
  LOCAL_DIR=/workspace/womd_v1_3/SMART_RAW_cache \
  bash /tmp/download_smart_raw_cache_from_nubes.sh \
  2>&1 | tee /workspace/womd_v1_3/SMART_RAW_cache_download.log
"
'
```

진행 상황은 아래처럼 확인한다.

```bash
kubectl exec -n p-pnc testa -c main -- tmux capture-pane -pt smart-raw-cache-download -S -80
kubectl exec -n p-pnc testa -c main -- bash -lc 'find /workspace/womd_v1_3/SMART_RAW_cache -type f | wc -l'
```

다운로드가 중간에 끊기면 같은 명령을 다시 실행하면 된다. 스크립트는 기본적으로 Nubes의 원격 파일 목록과 로컬 파일 수를 비교하고, `nubescli dir-download -s -j 96`으로 이미 존재하는 파일을 건너뛴다. `SKIP_REMOTE_LIST=1`을 쓰면 원격 목록 생성을 생략하고 바로 `dir-download`를 실행하므로 큰 cache를 처음 내려받을 때 더 빠르다.

### testa/testaa A100x4x2 SMART pretrain

`SMART` 브랜치의 기본 pretrain recipe를 유지하면서, 이미 떠 있는 `testa` 4 A100 + `testaa` 4 A100에서 멀티 노드 학습을 시작하려면 아래 wrapper를 사용한다. 이 launcher는 `kubectl exec`와 pod 내부 tmux만 사용하며, pod를 만들거나 삭제하거나 재시작하지 않는다.

```bash
bash scripts/start_smart_a100x4x2_testa_pretrain.sh
```

기본 실행 조건은 아래와 같다.

| 항목 | 기본값 |
|---|---|
| branch | `SMART` |
| pods | `testa testaa` |
| GPU | pod당 4개, 총 8 A100 |
| cache root | `/workspace/womd_v1_3/SMART_RAW_cache` |
| experiment | `pre_bc_a100x4x2` |
| action | `fit` |
| train batch size | per-GPU `10`, effective `80` |
| validation batch size | per-GPU `12` |
| max epochs | `64` |
| precision | `32-true` |
| gradient clipping | `0.5` |
| validation 주기 | `check_val_every_n_epoch=16` |
| closed-loop rollout 수 | `32` |
| validation rollout 후보 폭 | `top_k=48` |
| fit-time fast scorer | Waymo Sim Agents 2025 only |
| scorer scene 수 | 약 `1680` scenes |
| W&B | online, project `SMART-FLOW`, entity `jksg01019-naver-labs` |

실험 이름은 기본적으로 아래 형식으로 생성된다.

```text
smart_pretrain_a100x4x2_smart_raw_fast_rmm_<YYYYMMDD_HHMMSS>
```

고정 이름이나 batch를 쓰려면 환경 변수로 넘긴다.

```bash
TASK_NAME=smart_pretrain_a100x4x2_smart_raw_fast_rmm_probe \
TRAIN_BATCH_SIZE=10 \
bash scripts/start_smart_a100x4x2_testa_pretrain.sh
```

짧은 smoke test는 아래처럼 실행한다.

```bash
TASK_NAME=smart_pretrain_a100x4x2_smoke \
LIMIT_TRAIN_BATCHES=2 \
LIMIT_VAL_BATCHES=0 \
MAX_EPOCHS=1 \
WANDB_MODE=offline \
bash scripts/start_smart_a100x4x2_testa_pretrain.sh
```

진행 확인:

```bash
kubectl exec -it -n p-pnc testa -c main -- tmux attach -t catk-smart-a100x4x2-pretrain
kubectl exec -n p-pnc testa -c main -- tmux capture-pane -pt catk-smart-a100x4x2-pretrain -S -80
```

학습 session만 우아하게 중단하고 pod는 유지하려면:

```bash
for pod in testa testaa; do
  kubectl exec -n p-pnc "$pod" -c main -- \
    tmux send-keys -t catk-smart-a100x4x2-pretrain C-c
done
```

### testa/testaa A100x4x2 Waymo Sim Agents 2025 제출

학습이 끝난 checkpoint로 full validation set 또는 full test set에 대해 closed-loop rollout을 만들고, Waymo Sim Agents 2025 제출 archive를 만든 뒤 웹페이지에 자동 업로드하려면 아래 wrapper를 사용한다. 두 스크립트 모두 기존 `testa` / `testaa` pod 안에서만 tmux를 띄우며, pod를 만들거나 삭제하거나 재시작하지 않는다.

validation set 제출:

```bash
CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/<run>/epoch_last.ckpt \
TASK_NAME=smart_waymo_val_epochXXX_a100x4x2 \
bash scripts/start_smart_a100x4x2_testa_waymo_val_submission.sh
```

test set 제출:

```bash
CKPT_PATH=/mnt/nuplan/projects/catk/checkpoints/<run>/epoch_last.ckpt \
TASK_NAME=smart_waymo_test_epochXXX_a100x4x2 \
bash scripts/start_smart_a100x4x2_testa_waymo_test_submission.sh
```

기본 실행 조건은 아래와 같다.

| 항목 | validation 제출 | test 제출 |
|---|---|---|
| action | `validate` | `test` |
| experiment | `sim_agents_sub` | `sim_agents_sub` |
| branch | `SMART` | `SMART` |
| pods | `testa testaa` | `testa testaa` |
| GPU | A100 8개 | A100 8개 |
| cache root | `/workspace/womd_v1_3/SMART_RAW_cache` | `/workspace/womd_v1_3/SMART_RAW_cache` |
| rollout 수 | `32` | `32` |
| rollout 후보 폭 | `top_k=48` | `top_k=48` |
| metric path | Waymo Sim Agents 2025 | Waymo Sim Agents 2025 |
| output | `sim_agents_2025_submission.tar.gz` | `sim_agents_2025_submission.tar.gz` |
| auto upload | enabled | enabled |

자동 업로드에는 Waymo 로그인 상태 파일이 필요하다. 기본 위치는 pod 안의 project root 기준 아래 경로다.

```text
secrets/waymo/waymo_storage_state.json
```

GUI가 있는 머신에서 로그인 상태를 만들 때는 아래 명령을 사용한다.

```bash
python scripts/waymo_save_storage_state.py --browser-channel chrome
```

다른 위치에 저장된 로그인 상태 파일을 쓰려면 실행 시 명시한다.

```bash
WAYMO_STORAGE_STATE_PATH=/path/to/waymo_storage_state.json \
CKPT_PATH=/path/to/epoch_last.ckpt \
bash scripts/start_smart_a100x4x2_testa_waymo_val_submission.sh
```

`CKPT_PATH`가 master pod인 `testa`에만 있으면 launcher가 같은 path로 `testaa`에 복사하고 파일 크기를 확인한 뒤 DDP 실행을 시작한다. launcher는 한 번 정한 `RUN_ID`를 모든 rank에 공통으로 전달해서 모든 shard가 같은 `logs/<TASK_NAME>/runs/<RUN_ID>/` 아래에 저장되도록 강제한다. `testa`와 `testaa`는 같은 경로 문자열을 쓰더라도 실제 파일시스템이 공유되지 않을 수 있으므로, 제출 shard는 기본적으로 `testaa`에서 `testa`로 스트리밍 수집된다.

rollout은 끝났지만 shard 수집, archive 생성, 업로드 단계만 실패했다면 rollout을 다시 돌리지 않고 아래 복구 스크립트를 쓴다.

```bash
python scripts/finalize_smart_a100x4x2_testa_waymo_submission.py \
  --run-dir /mnt/nuplan/projects/catk/logs/<TASK_NAME>/runs/<RUN_ID> \
  --upload
```

이미 archive까지 만들어졌고 업로드만 재시도하려면:

```bash
python scripts/finalize_smart_a100x4x2_testa_waymo_submission.py \
  --run-dir /mnt/nuplan/projects/catk/logs/<TASK_NAME>/runs/<RUN_ID> \
  --skip-copy --skip-archive --upload
```

이 branch에서는 legacy metric/submission 경로를 지원하지 않는다. validation과 submission 관련 metric은 Waymo Sim Agents 2025 경로만 사용한다.

## Run the code
In the scripts, we provide
- [scripts/train.sh](scripts/train.sh) for training and fine-tuning.
- [scripts/local_val.sh](scripts/local_val.sh) for local validation.
- [scripts/sim_agents_sub.sh](scripts/sim_agents_sub.sh) for packing Waymo Sim Agents 2025 submission files.

The default script runs with single GPU. We use DDP for multi GPU training and validation, and the codes are also found in the bash scripts.
To reproduce our final results, you should follow the following steps
1. Use [scripts/train.sh](scripts/train.sh) with the [BC pre-training config](configs/experiment/pre_bc.yaml) to pre-train the SMART-tiny 7M model.
2. Use [scripts/train.sh](scripts/train.sh) with the [CLSFT with CAT-K config](configs/experiment/clsft.yaml) to fine-tune the SMART-tiny model pre-trained in step 1.
3. Use [scripts/sim_agents_sub.sh](scripts/sim_agents_sub.sh) to pack the submission file for `validate` or `test` split. This branch supports the Waymo Sim Agents 2025 metric/submission path only.
4. Alternatively, you can do local validation with [scripts/local_val.sh](scripts/local_val.sh).

For Gaussian Mixture Model (GMM) based ego policy, the procedure is similar, just use the following configs
- [BC pre-training config for GMM-based ego policy](configs/experiment/ego_gmm_pre_bc.yaml)
- [CLSFT with CAT-K config for GMM-based ego policy](configs/experiment/ego_gmm_clsft.yaml)
- [Local validation config for GMM-based ego policy](configs/experiment/ego_gmm_local_val.yaml)
- There is no submission option for ego-policy.

## Performance

This branch is configured for Waymo Sim Agents 2025 validation/submission. Legacy metric and submission code paths are intentionally not supported.

## Ablation configs

Please refer to [docs/ablation_models.md](docs/ablation_models.md) for the configurations of ablation models.
Specifically you will find the data augmentation methods used by [SMART](https://arxiv.org/abs/2207.05844) and [Trajeglish](https://arxiv.org/abs/2312.04535).

## Acknowledgement

Our code is based on [SMART](https://github.com/rainmaker22/SMART). We appreciate them for the valuable open-source code! Please don't forget to cite their amazing work as well!
