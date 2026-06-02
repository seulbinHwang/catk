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
- **Top on the WOSAC Leaderboard 2024**: With the Waymo Challenges 2025 coming up, the WOSAC 2024 leaderboard is now closed and our method remains in the 1st place.

Feb. 2025
- **Paper accepted at CVPR 2025:** Cheers!

- **Model checkpoints for WOSAC:** You can obtain the checkpoints for our WOSAC submission (SMART-tiny-CLSFT) by sending an email to Zhejun (zhejun.zhang94@gmail.com). In accordance with Waymo's terms, you must attach a screenshot showing that you are registered and logged into the [My Submissions](https://waymo.com/open/challenges/submissions) page of the Waymo Open Dataset.

- **SMART-mini and SMART-nano:** SMART-tiny with 7M parameters requires training on 8x A100 for a few days, which may be unaffordable in some cases. To address this, we have added config files for two smaller model, [smart_mini_3M.yaml](configs/model/smart_mini_3M.yaml) and [smart_nano_1M.yaml](configs/model/smart_nano_1M.yaml). Specifically, SMART-nano-1M can be trained on a single A100, but its performance is significantly worse. After pre-training and CAT-K fine-tuning, we achieved an RMM of 0.74 with SMART-nano-1M, which is 0.03 lower than that of SMART-tiny-7M. 

Jan. 2025
- **SoTA performance on WOSAC:** CAT-K is now rank #1 on the [WOSAC leaderboard](https://waymo.com/open/challenges/2024/sim-agents/)! We resolved an issue in the agent token vocabulary, and now our fine-tuned model achieves an RMM of **0.7702**. Even our reproduced SMART-tiny-7M (not published on the leaderboard, trained only for 32 epochs via BC) achieves an RMM of **0.7671**, which is comparable to the current second-place method. Reproducing our results should be straightforward. Give it a try!

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

## Run the code
In the scripts, we provide
- [scripts/train.sh](scripts/train.sh) for training and fine-tuning.
- [scripts/local_val.sh](scripts/local_val.sh) for local validation.
- [scripts/wosac_sub.sh](scripts/wosac_sub.sh) for packing submission files.

The default script runs with single GPU. We use DDP for multi GPU training and validation, and the codes are also found in the bash scripts.
To reproduce our final results, you should follow the following steps
1. Use [scripts/train.sh](scripts/train.sh) with the [BC pre-training config](configs/experiment/pre_bc.yaml) to pre-train the SMART-tiny 7M model.
2. Use [scripts/train.sh](scripts/train.sh) with the [CLSFT with CAT-K config](configs/experiment/clsft.yaml) to fine-tune the SMART-tiny model pre-trained in step 1.
3. Use [scripts/wosac_sub.sh](scripts/wosac_sub.sh) to pack the submission fille for `validate` or `test` split. Upload the `wosac_submission.tar.gz` file located in `logs` folder to the [WOSAC leaderboard](https://waymo.com/open/challenges/2024/sim-agents/) such that you can evaluate the model fine-tuned in step 2 on the WOSAC leaderboard.
4. Alternatively, you can do local validation with [scripts/local_val.sh](scripts/local_val.sh).

For Gaussian Mixture Model (GMM) based ego policy, the procedure is similar, just use the following configs
- [BC pre-training config for GMM-based ego policy](configs/experiment/ego_gmm_pre_bc.yaml)
- [CLSFT with CAT-K config for GMM-based ego policy](configs/experiment/ego_gmm_clsft.yaml)
- [Local validation config for GMM-based ego policy](configs/experiment/ego_gmm_local_val.yaml)
- There is no submission option for ego-policy.

## Performance

The submission of our CAT-K fine-tuned SMART to the [WOSAC Leaderboard](https://waymo.com/open/challenges/2024/sim-agents/) is found [here](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731338655639000/).
The submission of our reproduced SMART to the test split is found [here](https://waymo.com/open/challenges/sim-agents/results/5ea7a3eb-7337/1731391949275000/), note that it is not published to the leaderboard.

## Ablation configs

Please refer to [docs/ablation_models.md](docs/ablation_models.md) for the configurations of ablation models.
Specifically you will find the data augmentation methods used by [SMART](https://arxiv.org/abs/2207.05844) and [Trajeglish](https://arxiv.org/abs/2312.04535).

## Acknowledgement

Our code is based on [SMART](https://github.com/rainmaker22/SMART). We appreciate them for the valuable open-source code! Please don't forget to cite their amazing work as well!
