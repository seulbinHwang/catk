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

### 공정 비교용 SMART NTP 학습 agent selection

KFM 계열 실험과 SMART NTP pretrain을 공정하게 비교할 때는 학습 손실을 받는
agent 집합도 맞춰야 한다. 이를 위해 `configs/experiment/pre_bc.yaml`은
`data.train_use_eval_agent_selection: true`를 켠다.

이 설정이 켜진 SMART NTP pretrain은 학습 전용 150m 거리 clipping, 현재 100m
조건, 미래 valid 길이 조건, 그리고 scenario당 최대 32개 target random selection을
적용하지 않는다. 대신 validation과 같은 no-op agent transform을 사용하고,
별도 train mask가 없으면 모든 valid target에 대해 loss를 계산한다. 즉, 방법론
비교에서 "KFM은 넓은 agent 집합을 학습하고 SMART NTP는 near-ego 일부 agent만
학습한다"는 차이가 생기지 않도록 맞춘다.

기본 data config인 `configs/data/waymo.yaml`의 기본값은
`train_use_eval_agent_selection: false`이므로, 별도 override가 없는 기존 학습
동작은 유지된다. 또한 target builder는 PyG `BaseTransform`의 현재 실행 형식에
맞게 구현되어 있어, true/false 양쪽 train selection mode 모두 datamodule 생성
단계에서 바로 사용할 수 있다.

공정 비교에서는 어느 epoch의 가중치를 비교하는지도 맞춰야 한다.
`configs/experiment/pre_bc.yaml`은 KFM pretrain 설정과 같이
`val_closed/sim_agents_2025/realism_meta_metric`을 checkpoint monitor로 사용하고
`mode: max`를 적용한다. 따라서 SMART NTP pretrain도 마지막 epoch 가중치가 아니라
closed-loop validation realism 점수가 가장 높았던 가중치를 best checkpoint로
저장한다.

### WOSAC submission 실행 시 fast metric 비활성화

`configs/experiment/wosac_sub.yaml`은 제출 파일 생성 전용 설정이다. 이 모드에서는
WOSAC submission protobuf를 저장하는 것이 목적이고, validation 중 fast Sim Agents
metric을 따로 누적해 로깅할 필요가 없다. 따라서 해당 config는
`n_batch_sim_agents_metric: 0`과 `scorer_scene_num: 0`을 명시해서 local fast metric
계산을 끈다.

SMART 모델도 submission 모드에서는 scorer scene 수 자동 조정을 적용하지 않는다.
이렇게 하면 제출 파일의 내용은 유지하면서, 로깅되지 않을 metric state를 만들기
위해 앞쪽 validation batch를 불필요하게 평가하는 일을 피할 수 있다.

### Dynamic traffic-light staleness for SMART baselines

The SMART token baseline now uses the same traffic-light input semantics as the
control-space flow experiments used for method comparison. Traffic-light state is
no longer embedded as a static map-token feature. The map encoder keeps the
current observed light state only as metadata, and the agent-to-lane attention
relation receives:

- the current observed traffic-light state for that lane, and
- a normalized staleness scalar, defined as `prediction_time - observed_light_time`.

The scalar is clipped to `[-1s, 6s]` and divided by `6s`. Map elements without an
observed light keep a zero staleness value, while observed `UNKNOWN` lights still
carry the elapsed-time value. This keeps the input meaning as “this lane was
observed with state S Δt seconds ago” rather than treating traffic lights as
permanent map attributes.

During closed-loop SMART rollout, the first predicted 0.5s block sees the current
light at `0s` staleness, and later blocks use `0.5s`, `1.0s`, ... staleness. The
cache builder also checks that WOMD scenarios use the standard current raw step
`10`, so the observed-light timestamp and model staleness convention stay aligned.

SMART closed-loop validation/test runs all `n_rollout_closed_val` rollouts as one
rollout-major batch. The map encoder is evaluated once, then `pt_token`,
`position`, `orientation`, `batch`, and `light_type` are expanded together with
rollout-specific batch offsets. Agent tensors are expanded in the same rollout
order, and outputs are reshaped back to `[agent, rollout, time, ...]` before
metrics, WOSAC submission, and visualization code see them. This keeps the public
validation/test result interface unchanged while avoiding repeated inference
calls for each rollout.

Stochastic SMART validation/test sampling uses rollout/scenario-specific seeds,
derived from `validation_closed_seed`, `scenario_id`, and rollout index. Each
expanded scenario owns a persistent `torch.Generator` during the closed-loop
token rollout, so `topk_prob` sampling does not depend on whether rollouts are
executed one by one or as a larger rollout batch.

### Motion missingness features for SMART baselines

The SMART next-token baseline now exposes motion missingness to the agent
context in the same way as the kinematic-control flow experiments. Each agent
motion feature is `[coarse displacement norm, heading-relative displacement
angle, motion_valid]`. Invalid coarse displacements are zeroed, and
`motion_valid=0` tells the model that the zero displacement came from missing
motion rather than a real stop.

The agent-to-agent relation feature also includes the sender/receiver relative
coarse motion in the receiver frame plus a relative-motion validity bit. When
either side has missing motion, the relative motion channels are zeroed and the
validity bit is `0`.

This does not change SMART token validity or loss masking. SMART still treats a
0.5s token as valid when the two coarse endpoint tokens are valid. The
control-flow branch may still use stricter fine-step segment validity; this
change only aligns the available missingness information in the shared context
features. Because the embedding input dimensions changed, SMART checkpoints
from before this change are not shape-compatible with fresh pretraining runs.

### WOSAC-CPD / WOSAC-CES distribution metrics

Closed-loop validation and WOSAC submission export now also compute distribution metrics from the 10Hz rollouts that the model already generated. No extra rollout is created only for these metrics.

- `val_closed/WOSAC-CPD/value`: conditional pairwise diversity among rollouts from the same scenario. Higher means the rollouts are more diverse.
- `val_closed/WOSAC-CES/value`: conditional Energy Score-style distribution metric. It is computed only when validation GT future is available. Lower is better.
- `test/WOSAC-CPD/value`: CPD measured during test submission export. Test split has no GT future, so CES is not reported.
- `*/WOSAC-CPD/DPR`: diversity preservation ratio. It is logged only when `model.model_config.wosac_cpd_reference` is set to a positive pretrain CPD value.

The two metrics use the same WOSAC joint rollout distance. Validation CPD/CES normalize each agent type by an automatically computed future-motion scale from validation GT. The scale is computed inside the metric and is not a user-chosen threshold.

Example for reporting CAT-K fine-tuning diversity preservation:

```bash
python -m src.run \
  experiment=clsft \
  ckpt_path=/path/to/catk.ckpt \
  model.model_config.wosac_cpd_reference=<SMART_PRETRAIN_CPD>
```

If `n_rollout_closed_val=32`, the metrics use those 32 rollouts. If it is changed to 16, the metrics use the already generated 16 rollouts.

### RoaD fine-tuning

This repository also supports RoaD-style closed-loop fine-tuning through:

```text
configs/experiment/road_clsft.yaml
```

RoaD fine-tuning is implemented as an epoch-local generated dataset pipeline:

- The original WOMD training pickle cache is never overwritten.
- At the beginning of each fine-tuning epoch, the latest SMART model generates a temporary RoaD cache from the original training cache.
- Each scenario generates 3 independent RoaD rollouts by default.
- The training dataset keeps the same effective length as the original scenario count by uniformly sampling one of the 3 generated rollouts for each scenario.
- The temporary cache is regenerated every epoch and deleted after use by default.

Default RoaD settings:

| Item | Value |
| --- | ---: |
| config | `road_clsft` |
| action | `road_finetune` |
| learning rate | `5e-5` |
| label smoothing | `0.0` |
| rollouts per scenario | `3` |
| candidate policy | Sample-K |
| candidate count | `64` |
| sampling temperature | `0.8` |
| candidate selection | closest sampled token to the expert next state |
| in-step CAT-K rollout during RoaD training | off |

Run RoaD fine-tuning with an explicit SMART BC checkpoint:

```bash
torchrun -m src.run \
  experiment=road_clsft \
  ckpt_path=/path/to/SMART_BC_PRETRAINED.ckpt \
  task_name=road_clsft
```

or use:

```bash
bash scripts/road_train.sh
```

For DDP training, use the DDP block in `scripts/road_train.sh`. Rank 0 generates the epoch-local RoaD cache under `${paths.output_dir}/road_cache/epoch_XXX`, then all ranks synchronize and read that cache. Set `road.delete_after_use=false` only when debugging generated data.

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
