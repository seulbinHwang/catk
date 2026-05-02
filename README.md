# CAT-K Flow Matching

이 저장소는 **flow matching 학습/추론/평가 전용**으로 정리된 버전입니다.  
기본 실행 경로와 문서, 스크립트는 모두 `smart_flow` 계열만 사용하며 CrossEntropy 기반 next-token 경로는 제거했습니다.  
현재 closed-loop local 평가와 제출 export는 **WOSAC 2025 / Waymo 2025 Sim Agents 기준**만 사용합니다.

- 기존 SMART의 map/context trunk를 그대로 재사용하고, agent 쪽만 flow decoder로 바꿔 scene-context 품질을 유지합니다.
- `FlowTokenProcessor`는 14-slot context pack과 13개 anchor를 만들되, 
- **context 위치/방향과 flow target 원점은 token-restored 상태가 아니라 실제 coarse 상태**를 사용합니다.
- agent coarse token id는 **마지막 점 1개가 아니라 0.5초 전체 6개 점 사각형 경로**를 기준으로 매칭합니다.
- `trajectory_token_veh/ped/cyc` 임베딩은 마지막 contour 1개 대신 
- **`agent_token_all_*` 전체 chunk(6 x 4 x 2)** 를 그대로 펼쳐 사용합니다.
- `HierarchicalFlowDecoder`와 `FlowODE`가 local normalized future를 직접 복원해 discrete token id보다 trajectory geometry를 더 부드럽게 모델링합니다.
- closed-loop inference는 0.5초씩 commit 하며 `pred_traj_10hz`, `pred_head_10hz`, `pred_z_10hz`를 바로 내보내 2025 Sim Agents rollout proto와 바로 연결됩니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm` 이 기본값이며, 
- 이때 외부로 내보내는 `pred_traj_10hz`, `pred_head_10hz`는 raw FM 출력 그대로 유지합니다.
- `model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk` 를 쓰면 
- `retokenize`로 고른 token의 0.5초 chunk를 **외부 rollout 10Hz 출력에만** 반영합니다.
- 내부 closed-loop context는 계속 실제 FM commit 상태를 유지합니다.
- `model.model_config.decoder.use_stop_motion=true` 를 켜면 current + 0.1/0.2/0.3/0.4/0.5초
  6점 경로를 motion token으로 다시 보고, **stop token** 과 일치하는 agent의 다음 0.5초 chunk를
  완전히 고정합니다. 이 stop gate는 vehicle / pedestrian / bicycle 모두에 적용됩니다.
- 이 stop-motion 토큰 매칭은 **실제 actor box 크기 대신 class별 고정 토큰 박스**를 사용합니다.
  vehicle은 `2.0 x 4.8`, pedestrian은 `1.0 x 1.0`, bicycle은 `1.0 x 2.0` 입니다.
- `model.model_config.decoder.use_lqr=true` 를 켜면 stop gate를 통과한 vehicle / bicycle에만
  curvature-domain LQR + kinematic bicycle commit bridge를 적용합니다. 이 모드에서는 2초 FM
  미래를 preview로 보되, 실제 반영은 항상 다음 0.5초 / 5점만 실행합니다.
- LQR bridge는 최근 실제 10Hz 6점 history로 현재 speed / yaw-rate / curvature를 잡고,
  `draft_physics.py`의 차종별 속도, 가감속, yaw-rate, 횡가속, 최소 선회 반경 제한을 같이 씁니다.
- wheelbase가 없는 WOMD multi-agent 특성을 고려해 steering angle 대신 **curvature를 제어 입력**
  으로 쓰는 kinematic bicycle 계열 적분을 쓰며, class별 envelope로 곡률과 곡률 변화율을 한 번 더
  clip 합니다.
- DRaFT physics 경로에는 NaN 방지 가드가 들어 있습니다.
- heading 2-vector와 pedestrian velocity 2-vector는 raw `atan2` 대신 safe angle 복원으로 처리해
  `(0, 0)` 또는 near-zero vector backward에서 gradient NaN이 나지 않도록 막습니다.
- `sample_open_loop_future` 결과나 physics loss 출력이 non-finite면 해당 batch의 draft loss를 0으로
  처리해 flow decoder 전체를 오염시키지 않게 합니다.
- 학습 중에는 non-finite parameter, `fm_loss`, `total_loss`, 비-AMP gradient를 fail-fast로 감지해
  NaN checkpoint가 조용히 저장되지 않도록 즉시 중단합니다. 단, `16-mixed` AMP의 scaled gradient
  overflow는 PyTorch `GradScaler`가 optimizer step을 skip하고 scale을 낮추도록 맡깁니다.
- closed-loop local 평가는 `SimAgentsMetrics`가 Waymo 공식 2025 scorer를 그대로 호출해 `val_closed/sim_agents_2025/*`와 `val_closed/sim_agents_2025_mean/*`를 기록합니다.
- submission export는 `SimAgentsSubmission`이 2025 submission shard와 `sim_agents_2025_submission.tar.gz`를 생성합니다.
- 설치 시점에 official 2025 scorer와 `traffic_light_violation` 관련 2025 필드가 실제로 있는지 바로 검증합니다.

### TODO: Motion Missingness Feature

- 현재 flow encoder는 invalid context step과 valid context step 사이의 motion을 `0`으로 두어 `(0, 0)` padding에서 global 좌표로 튀는 가짜 초대형 motion을 막습니다.
- 장기적으로는 `motion value = 0`에 더해 `motion_valid = false` 같은 별도 feature를 추가해, 실제 정지 agent와 이전 motion을 알 수 없는 agent를 구분하는 편이 더 완전합니다.
- 다만 이 방식은 motion embedding input dimension을 바꾸므로 기존 pretrained checkpoint와 바로 호환되지 않습니다. 새 pretraining을 처음부터 설계할 때 검토할 TODO로 둡니다.

### Closed-loop Retokenize Rule

- `retokenize` 자체는 **현재 실제 coarse 상태 + 이번 0.5초 raw FM commit 5점**을 합친 6개 점 경로를 기준으로 
- 다음 token id를 다시 고릅니다.
- `pos_window`, `head_window`, `coarse_pos/head`, 그리고 다음 step motion feature는 
- 모두 **token bank 복원값이 아니라 실제 FM commit의 마지막 상태** 기준으로 갱신합니다.
- 기본값 `raw_fm` 에서는 `pred_traj_10hz`, `pred_head_10hz`를 raw FM 출력 그대로 유지합니다. 
- 따라서 WOSAC metric, submission proto, video visualization은 
- post-process된 token endpoint가 아니라 네트워크가 직접 낸 10Hz trajectory를 봅니다.
- `matched_token_chunk` 에서는 같은 6점 경로 매칭으로 고른 token chunk가 외부 rollout에도 반영됩니다. 
- 다만 내부 closed-loop context는 계속 실제 상태를 유지합니다.
- `use_lqr=true` 를 켠 경우에도 `retokenize`와 내부 문맥 갱신은
  항상 실행된 5개 fine 상태를 기준으로 이뤄집니다.
- 같은 모드에서 `matched_token_chunk`를 써도 vehicle / bicycle의 외부 10Hz 출력은
  token chunk로 다시 덮지 않고 실제 실행 chunk를 유지합니다. pedestrian만 기존 방식대로
  token chunk export를 유지합니다.



### DRaFT Top-K Feasibility Loss

- DRaFT physics loss는 기본적으로 2초 미래 20개 시점의 물리 위반을 한 번 계산해서 시간 평균과 시간축 상위-K 평균을 동시에 얻고, 둘을 절반씩 섞어 최종 손실로 씁니다.
- `model.model_config.draft.physics.topk_violation_k` 가 K 입니다. K 가 T (=20) 이상이면 상위-K 가 시간 평균과 같아져 단일 mean 경로로만 동작합니다.
- 최종 physics loss는 `0.5 * (시간 평균 loss + 상위-K 위반 loss)` 입니다.
- **기본값은 `topk_violation_k=4`** 입니다. 한두 프레임의 급가속, 급회전, 순간 점프가 평균에 묻히지 않도록 곧바로 강조되도록 켠 default 입니다 (이전에는 `K=20` no-op 이 기본이라 사용자가 명시적으로 줄여야 활성화됐습니다).
- 더 부드럽게 평균 위주로 학습하고 싶으면 `topk_violation_k=8` 이나 `topk_violation_k=20` (mean-only) 으로 늘리세요. 더 강한 worst-step 집중을 원하면 `topk_violation_k=2`/`1` 도 가능합니다.
- 시점별 위반은 클래스 당 한 번만 계산되어 mean 과 상위-K 집계에 동시에 사용됩니다. 이전 구현이 가졌던 mean 경로 + topk 경로의 이중 forward 비용이 없습니다.
- 이 설정은 `draft.max_weight`, sampling step, sampling method, backprop_last_k, batch size, learning rate를 바꾸지 않습니다.

예시:

```bash
# 기본값 (K=4) - mean + 상위-4 worst-step blend
... model.model_config.draft.physics.topk_violation_k=4

# Mean-only (이전 default 와 동일)
... model.model_config.draft.physics.topk_violation_k=20
```

### DRaFT Soft-Limit Ratio

- DRaFT physics hard loss는 물리량을 한계값으로 나눈 뒤 `1.0`을 넘은 초과분에 벌점을 줍니다.
- `model.model_config.draft.physics.soft_limit_ratio` 는 이 벌점 시작점을 낮추는 값입니다.
- 물리량 `z_t`, hard limit `z_max`, soft-limit 비율 `rho` 에 대해 손실은 `max(0, z_t / z_max - rho)^2` 입니다.
- 기본값은 `soft_limit_ratio=1.0` 입니다. 이 값이면 기존 hard-only 방식과 동일하게 동작합니다.
- `soft_limit_ratio=0.85` 이면 hard limit의 85%를 넘는 순간부터 벌점이 생깁니다.
- 보수적인 시작점은 `0.85`, 더 강한 시작점은 `0.75`를 권장합니다.
- 너무 낮추면 실제로 가능한 빠른 움직임까지 억제할 수 있습니다.
- 이 설정은 `draft.max_weight`, sampling step, sampling method, backprop_last_k, batch size, learning rate를 바꾸지 않습니다.

예시:

```bash
# 기존 hard-only와 동일
... model.model_config.draft.physics.soft_limit_ratio=1.0

# 보수적 soft-limit
... model.model_config.draft.physics.soft_limit_ratio=0.85

# 더 강한 soft-limit
... model.model_config.draft.physics.soft_limit_ratio=0.75
```

### MLX Multi-Node V100x8 DRaFT Fine-Tuning

V100 8장짜리 Pod 여러 개로 **하나의 DRaFT fine-tuning**을 돌릴 수 있습니다. 전체 GPU 수는 `8 x worker 수`이고, 모든 worker Pod에서 같은 repo branch, 같은 cache root, 같은 pretrain checkpoint 경로가 보여야 합니다.

관련 스크립트:

- `scripts/launch_mlx_static_pods_tmux.py`: 이미 떠 있는 고정 Pod에 tmux 세션을 만들고 static `torchrun`을 시작합니다.
- `scripts/launch_mlx_static_pods_bs_sweep.py`: 고정 Pod에서 `train_batch_size`를 키워 시도하고, CUDA OOM이면 batch size를 낮춰 재시작/재개합니다.
- `scripts/create_mlx_static_v100_pods.py`: `testsv` 같은 장기 실행 V100 Pod를 만들고, 스케줄링이 안 되면 CPU/memory request를 낮춰 재시도합니다.
- `scripts/prepare_mlx_static_pods_assets.py`: 각 static Pod 안에서 repo, conda env, SMART cache, W&B checkpoint를 준비하고 검증합니다.
- `scripts/render_mlx_finetune_pytorchjob.py`: 새 worker Pod를 만드는 Kubeflow `PyTorchJob` YAML을 생성합니다.
- `scripts/mlx_finetune_draft_flow_v100x8_multinode.sh`: 각 worker 안에서 실제 `torchrun`을 실행하는 공통 wrapper입니다.

#### Case 1. `testv`, `testvv` Pod가 이미 떠 있는 경우

이 경우가 가장 빠릅니다. kubectl이 되는 터미널에서 이 repo checkout으로 이동해 아래 명령을 한 번 실행하면, `testv`와 `testvv` 안에 같은 이름의 tmux session이 만들어지고 각각 rank 0/1로 학습이 시작됩니다. `MASTER_ADDR`는 `testv`의 Pod IP로 자동 설정됩니다. Pod 안 repo 위치는 기본값이 `/mnt/nuplan/projects/catk`이고, 다르면 `--project-root`로 바꾸면 됩니다.

`finetune_draft_flow_v100x8`의 기본 batch 설정은 실측 안정값인 `data.train_batch_size=36`, `trainer.accumulate_grad_batches=1`입니다. `testv` + `testvv` 2-node run에서는 effective batch가 `36 * 16 GPUs * 1 = 576`입니다.

V100x8 fit-time validation은 비교 공정성을 유지합니다. 이전 2-node run에서 Epoch 15 train은 정상 종료됐지만, `check_val_every_n_epoch=16`으로 시작된 closed-loop validation 중 공식 `sim_agents_2025` TensorFlow scorer가 오래 CPU를 점유했고, 한 worker가 먼저 죽은 뒤 다음 NCCL collective에서 전체 run이 abort된 적이 있습니다. 그래서 `finetune_draft_flow_v100x8`은 평가량을 줄이지 않고 `data.val_batch_size=4`, `model.model_config.n_rollout_closed_val=16`, `model.model_config.n_batch_sim_agents_metric=10`, `trainer.limit_val_batches=0.1`을 유지합니다. 안정성 조치는 `model.model_config.sim_agents_metric_workers=1`로 scorer를 rank마다 순차 실행하고, NCCL heartbeat timeout을 길게 두는 것입니다. 이 설정은 느릴 수 있지만, 평가하는 batch/rollout/scorer 수를 줄이지 않습니다.

먼저 짧은 smoke run을 권장합니다.

```bash
cd /path/to/catk

python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --learning-rate 2e-4 \
  --limit-train-batches 40 \
  --limit-val-batches 0 \
  --max-epochs 1 \
  --session catk-draft-ft \
  --replace
```

각 Pod의 tmux에 들어가면 위 pane에는 training stdout/tqdm이, 아래 pane에는 GPU 사용률 heartbeat가 보입니다.

```bash
kubectl exec -it -n p-pnc testv -c main -- tmux attach -t catk-draft-ft
kubectl exec -it -n p-pnc testvv -c main -- tmux attach -t catk-draft-ft
```

tmux에서 빠져나올 때는 `Ctrl-b d`를 누릅니다. 학습을 중단하려면:

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --container main \
  --session catk-draft-ft \
  --stop
```

smoke run이 정상 연결되면 full run은 batch 제한만 빼고 다시 시작합니다.

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --learning-rate 2e-4 \
  --session catk-draft-ft \
  --replace
```

기본적으로 런처는 각 Pod에서 `origin/<branch>`를 fetch/pull 합니다. Pod 안 repo에 실험용 로컬 수정이 있어서 pull하면 안 되는 경우에만 `--no-pull`을 붙이세요.

#### Case 1-A. `testv`, `testvv`에서 train_batch_size OOM sweep

`train_batch_size=36`, `accumulate_grad_batches=1`은 이제 V100x8 기본값입니다. 새 V100 박스나 다른 데이터 상태에서 OOM이 걱정되면, 같은 값에서 시작해 CUDA OOM이 날 때마다 `8`씩 낮추는 자동 sweep을 쓸 수 있습니다. 이 스크립트는 기존 static tmux launcher를 반복 호출합니다.

기본 동작:

- 첫 시도: `train_batch_size=36`, `accumulate_grad_batches=1`, `action=finetune`
- OOM 감지: `train_batch_size -= 8`
- 재개 checkpoint가 있으면: `action=fit`, `ckpt_path=<latest epoch_last.ckpt>`
- 재개 checkpoint가 아직 없으면: 더 작은 batch size로 pretrained checkpoint에서 epoch 0 재시도
- non-OOM 에러: batch size를 낮춰도 해결되지 않는 문제로 보고 중단

주의: `epoch_last.ckpt`는 train epoch이 끝난 뒤 저장됩니다. 따라서 mid-epoch OOM은 정확히 그 batch에서 재개하는 것이 아니라, **마지막으로 완료된 epoch checkpoint**에서 재개합니다. 0 epoch 중 OOM이 나서 checkpoint가 없으면 더 작은 batch size로 처음부터 다시 시작합니다.

현재 2-node 기본 run은 `36 * 16 GPUs * accumulate 1 = 576` effective batch입니다. sweep의 첫 시도도 같은 설정이고, OOM이 나는 경우에만 `28 -> 20 -> 12 -> 4` 순서로 낮춥니다.

실행 예시:

```bash
cd /path/to/catk

nohup python -u scripts/launch_mlx_static_pods_bs_sweep.py \
  --namespace p-pnc \
  --pods testv testvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --soft-limit-ratio 0.8 \
  --learning-rate 2e-4 \
  --start-batch-size 36 \
  --batch-step 8 \
  --min-batch-size 4 \
  --accumulate-grad-batches 1 \
  --task-name catk_draft_v100x8x2_soft_limit_ratio_0.8_bs_sweep \
  --session catk-draft-bs-sweep \
  > /tmp/catk_draft_v100x8x2_bs_sweep.log 2>&1 &

tail -f /tmp/catk_draft_v100x8x2_bs_sweep.log
```

현재 attempt의 training stdout/tqdm은 master Pod인 `testv`의 tmux에서 봅니다.

```bash
kubectl exec -it -n p-pnc testv -c main -- tmux attach -t catk-draft-bs-sweep
```

worker Pod인 `testvv`는 global rank 8-15만 갖기 때문에 Lightning progress bar가 기본적으로 출력되지 않습니다. 그래도 학습에는 참여 중이며 GPU heartbeat pane이나 `nvidia-smi`로 사용률을 확인할 수 있습니다.

validation 도중 worker가 죽어서 중단된 경우에는 master pod에 남아 있는 `logs/<task_name>/runs/<timestamp>/checkpoints/epoch_last.ckpt`를 양쪽 pod의 같은 경로로 복사한 뒤 `action=fit`으로 재개합니다. `epoch_last.ckpt`는 validation 시작 직전의 loop 상태를 담고 있어서, 이미 끝난 train epoch을 다시 돌지 않고 완료하지 못한 fit-time validation부터 이어갈 수 있습니다.

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --action fit \
  --ckpt-path /mnt/nuplan/projects/catk/checkpoints/<copied_epoch_last>.ckpt \
  --learning-rate 2e-4 \
  --soft-limit-ratio 0.8 \
  --train-batch-size 36 \
  --accumulate-grad-batches 1 \
  --task-name catk_draft_v100x8x2_soft_limit_ratio_0.8_valfix_resume \
  --session catk-draft-bs-sweep \
  --replace
```

8-GPU pod 하나와 7-GPU pod 하나처럼 GPU 수가 서로 다르면 일반 `torchrun --nproc_per_node 8 --nnodes 2`는 사용할 수 없습니다. 이때는 hetero static launcher를 사용합니다. 이 launcher는 각 GPU를 1-GPU logical node로 보고, `testv`의 8개 GPU와 `testvv`의 7개 GPU를 합쳐 `world_size=15`로 실행합니다.

validation 도중 죽은 run을 16 GPU에서 15 GPU로 바꿔 복구할 때는 먼저 같은 checkpoint로 validation-only를 통과시키는 편이 안전합니다. 15 GPU에서 16 GPU run과 같은 공식 scorer scene 수를 맞추려면 `n_batch_sim_agents_metric=11`, `n_scenario_sim_agents_metric=640`을 같이 줍니다. 이렇게 하면 16 GPU run의 `10 batch * 4 scene * 16 rank = 640 scene`과 같은 수만 공식 scorer에 들어갑니다.

```bash
python scripts/launch_mlx_hetero_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --nproc-per-pod 8 7 \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --action validate \
  --ckpt-path /mnt/nuplan/projects/catk/checkpoints/catk_draft_v100x8x2_soft_limit_ratio_0.8_bs_sweep_20260502_0055/epoch15_valfix_resume.ckpt \
  --experiment finetune_draft_flow_v100x8 \
  --learning-rate 2e-4 \
  --soft-limit-ratio 0.8 \
  --train-batch-size 36 \
  --accumulate-grad-batches 1 \
  --extra-hydra-overrides "model.model_config.n_batch_sim_agents_metric=11 model.model_config.n_scenario_sim_agents_metric=640" \
  --task-name catk_draft_v100x8x7_hetero15_soft_limit_ratio_0.8_valfix_validate \
  --session catk-draft-hetero15-validate \
  --master-port 29541 \
  --replace
```

validation이 성공하면 학습은 `action=fit`으로 다시 띄웁니다. world size가 16에서 15로 바뀌면 epoch 안의 batch 개수가 달라지므로, 이미 끝난 epoch 15 training을 다시 돌지 않도록 loop progress를 epoch-complete 상태로 맞춘 checkpoint를 쓰는 것이 안전합니다.

```bash
python scripts/launch_mlx_hetero_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testv testvv \
  --nproc-per-pod 8 7 \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --action fit \
  --ckpt-path /mnt/nuplan/projects/catk/checkpoints/catk_draft_v100x8x2_soft_limit_ratio_0.8_bs_sweep_20260502_0055/epoch15_valfix_resume_epoch_complete_for_15gpu.ckpt \
  --experiment finetune_draft_flow_v100x8 \
  --learning-rate 2e-4 \
  --soft-limit-ratio 0.8 \
  --train-batch-size 36 \
  --accumulate-grad-batches 1 \
  --extra-hydra-overrides "model.model_config.n_batch_sim_agents_metric=11 model.model_config.n_scenario_sim_agents_metric=640" \
  --task-name catk_draft_v100x8x7_hetero15_soft_limit_ratio_0.8_valfix_resume \
  --session catk-draft-hetero15-valfix \
  --replace
```

이 구성의 effective batch는 `36 * 15 * 1 = 540`입니다. 기존 16-GPU run의 `576`보다 약 6.25% 작지만, checkpoint의 optimizer/scheduler 상태는 그대로 복원되므로 validation 복구용으로는 이 차이가 가장 작은 현실적인 선택입니다.

#### Case 1-B. `testsv`, `testsvv`, `testsvvv`, `testsvvvv` V100x4 Pod 4개를 새로 만들어 쓰는 경우

V100 4장짜리 Pod 4개로 하나의 run을 돌릴 때는 전체 GPU가 `4 * 4 = 16`장입니다. 이 경우 `--nproc-per-node 4`를 주면 런처가 기본 experiment를 `finetune_draft_flow_v100x4`로 자동 선택합니다. 이 preset의 기본 batch 설정은 V100x8과 동일하게 `data.train_batch_size=36`, `trainer.accumulate_grad_batches=1`입니다.

```text
train_batch_size 36 * total_gpus 16 * accumulate_grad_batches 1 = 576
```

`soft_limit_ratio=0.9`는 `finetune_draft_flow_v100x4` 안의 기본값이지만, 명령줄에 명시해도 됩니다. 이 구성은 V100x8 Pod 2개 구성과 같은 effective batch `576`을 사용합니다.

1. Pod 생성

```bash
cd /path/to/catk

python scripts/create_mlx_static_v100_pods.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --gpu-count 4 \
  --zone private-v100-naverlabs-0
```

첫 profile은 `requests.cpu=32`, `requests.memory=128Gi`, `limits.memory=480Gi`, `nvidia.com/gpu=4`입니다. 스케줄링이 안 되면 같은 Pod 이름을 지웠다가 `96Gi/384Gi`, `64Gi/256Gi` 순서로 낮춰 재시도합니다. 이미 같은 이름의 Pod가 있으면 건드리지 않습니다. 처음부터 다시 만들려면 `--replace`를 붙입니다.

2. repo / conda env / cache / checkpoint 준비

```bash
python scripts/prepare_mlx_static_pods_assets.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --cache-source labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
  --artifact jksg01019-naver-labs/SMART-FLOW/epoch-last-4pxhrpv8:v70 \
  --ckpt-path /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --replace
```

이 스크립트는 각 Pod 안에 `catk-pod-prepare` tmux session을 만들고 병렬로 준비합니다. SMART cache 다운로드는 보통 오래 걸립니다. 중간 상태만 보려면:

```bash
python scripts/prepare_mlx_static_pods_assets.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --status-only
```

개별 로그를 보고 싶으면:

```bash
kubectl exec -it -n p-pnc testsv -c main -- tmux attach -t catk-pod-prepare
```

3. 짧은 multi-node smoke run

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --nproc-per-node 4 \
  --learning-rate 2e-4 \
  --limit-train-batches 40 \
  --limit-val-batches 0 \
  --max-epochs 1 \
  --task-name catk_draft_v100x4x4_soft_limit_ratio_0.9_bs36_acc1_smoke \
  --session catk-draft-v100x4x4 \
  --replace
```

progress bar는 master Pod인 `testsv`에서 봅니다.

```bash
kubectl exec -it -n p-pnc testsv -c main -- tmux attach -t catk-draft-v100x4x4
```

4. full run

smoke run 연결이 정상임을 확인한 뒤 같은 session 이름으로 full run을 다시 시작합니다.

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --nproc-per-node 4 \
  --learning-rate 2e-4 \
  --task-name catk_draft_v100x4x4_soft_limit_ratio_0.9_bs36_acc1 \
  --session catk-draft-v100x4x4 \
  --replace
```

중단:

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods testsv testsvv testsvvv testsvvvv \
  --container main \
  --session catk-draft-v100x4x4 \
  --stop
```

#### Case 1-C. `fv`, `fvv`, `fvvv`, `fvvvv`, `fvvvvv` V100x3 Pod 5개를 추가로 쓰는 경우

기존 `testv/testvv` 또는 `testsv/testsvv/testsvvv/testsvvvv` 실험을 그대로 둔 채, V100 3장짜리 Pod 5개를 새로 추가해서 별도 run을 돌리는 구성입니다. 전체 GPU는 `3 * 5 = 15`장이고, `--nproc-per-node 3`을 주면 런처가 기본 experiment를 `finetune_draft_flow_v100x3`로 자동 선택합니다.

```text
train_batch_size 36 * total_gpus 15 * accumulate_grad_batches 1 = 540
```

`finetune_draft_flow_v100x3`의 기본값은 `soft_limit_ratio=0.7`, `data.train_batch_size=36`, `trainer.accumulate_grad_batches=1`입니다. 이 실험은 기존 Pod를 삭제하지 않고 새 `fv*` Pod만 사용합니다.

1. Pod 생성

```bash
cd /path/to/catk

python scripts/create_mlx_static_v100_pods.py \
  --namespace p-pnc \
  --pods fv fvv fvvv fvvvv fvvvvv \
  --gpu-count 3 \
  --zone private-v100-naverlabs-0
```

첫 profile은 `requests.cpu=32`, `requests.memory=128Gi`, `limits.memory=480Gi`, `nvidia.com/gpu=3`입니다. 스케줄링이 안 되면 같은 `fv*` Pod 이름만 지웠다가 `96Gi/384Gi`, `64Gi/256Gi` 순서로 낮춰 재시도합니다. 기존 `test*` Pod는 건드리지 않습니다.

2. repo / conda env / cache / checkpoint 준비

```bash
python scripts/prepare_mlx_static_pods_assets.py \
  --namespace p-pnc \
  --pods fv fvv fvvv fvvvv fvvvvv \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --cache-source labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
  --artifact jksg01019-naver-labs/SMART-FLOW/epoch-last-4pxhrpv8:v70 \
  --ckpt-path /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --replace
```

중간 상태:

```bash
python scripts/prepare_mlx_static_pods_assets.py \
  --namespace p-pnc \
  --pods fv fvv fvvv fvvvv fvvvvv \
  --status-only
```

3. full run

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods fv fvv fvvv fvvvv fvvvvv \
  --container main \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --nproc-per-node 3 \
  --learning-rate 2e-4 \
  --task-name catk_draft_v100x3x5_soft_limit_ratio_0.7_bs36_acc1 \
  --session catk-draft-v100x3x5-soft07 \
  --replace
```

progress bar는 master Pod인 `fv`에서 봅니다.

```bash
kubectl exec -it -n p-pnc fv -c main -- tmux attach -t catk-draft-v100x3x5-soft07
```

중단:

```bash
python scripts/launch_mlx_static_pods_tmux.py \
  --namespace p-pnc \
  --pods fv fvv fvvv fvvvv fvvvvv \
  --container main \
  --session catk-draft-v100x3x5-soft07 \
  --stop
```

#### Case 2. 아직 worker Pod가 없는 경우

새 Pod를 만들어야 하면 Kubeflow `PyTorchJob`이 가장 깔끔합니다. PyTorchJob은 worker Pod 수, rendezvous endpoint, rank 관련 환경을 자동으로 맞춥니다. 이 방식은 Kubernetes Job lifecycle을 따르므로 tmux 대신 `kubectl logs`로 보는 것이 기본입니다. 꼭 tmux가 필요하면 먼저 장기 실행 Pod를 만든 뒤 Case 1의 static tmux launcher를 쓰세요.

먼저 짧은 PyTorchJob smoke run:

```bash
cd /path/to/catk

python scripts/render_mlx_finetune_pytorchjob.py \
  --workers 2 \
  --job-name catk-draft-v100x8x2-smoke \
  --namespace p-pnc \
  --zone private-v100-naverlabs-0 \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --learning-rate 2e-4 \
  --limit-train-batches 40 \
  --limit-val-batches 0 \
  --extra-hydra-overrides trainer.max_epochs=1 \
  --output /tmp/catk-draft-v100x8x2-smoke.yaml

kubectl apply -f /tmp/catk-draft-v100x8x2-smoke.yaml
```

상태와 로그 확인:

```bash
kubectl get pytorchjob catk-draft-v100x8x2-smoke -n p-pnc
kubectl get pods -n p-pnc | grep catk-draft-v100x8x2-smoke
kubectl logs -f catk-draft-v100x8x2-smoke-worker-0 -c pytorch -n p-pnc
```

full run:

```bash
python scripts/render_mlx_finetune_pytorchjob.py \
  --workers N \
  --job-name catk-draft-v100x8xN \
  --namespace p-pnc \
  --zone private-v100-naverlabs-0 \
  --branch semi_continuous_track_loss \
  --cache-root /workspace/womd_v1_3/SMART_cache \
  --pretrain-ckpt /mnt/nuplan/projects/catk/checkpoints/flow_semi_continuous_pretrain_all_target_h1006/4pxhrpv8_v70_e64_step259776/epoch_last.ckpt \
  --learning-rate 2e-4 \
  --output /tmp/catk-draft-v100x8xN.yaml

kubectl apply -f /tmp/catk-draft-v100x8xN.yaml
```

`--learning-rate auto`를 쓰면 V100x8 단일 노드 기준 `2e-4`에서 worker 수만큼 linear scaling합니다. 보수적으로 시작하려면 위 예시처럼 `--learning-rate 2e-4`로 고정하세요.

#### V100 `16-mixed`와 non-finite gradient 처리

V100은 BF16을 네이티브로 쓰기 어려우므로 fine-tuning preset은 기본적으로 `trainer.precision=16-mixed`를 씁니다. `16-mixed`에서는 Lightning/PyTorch가 `GradScaler`로 loss를 크게 키운 뒤 backward를 수행합니다. 이때 일부 gradient가 FP16 범위를 넘어 `inf`가 될 수 있지만, 이는 AMP가 예상하는 recoverable overflow입니다.

중요한 순서는 아래와 같습니다.

```text
scale(loss) -> backward -> on_after_backward hook -> unscale gradients -> optimizer step or skip -> scale update
```

따라서 `on_after_backward`에서 scaled gradient를 즉시 fail-fast하면, `GradScaler`가 원래 하려던 "이번 step skip + scale 낮추기"가 실행되기 전에 학습이 종료됩니다. 이 저장소는 `16-mixed`일 때 backward 직후 gradient fail-fast를 건너뛰고, AMP overflow 처리는 `GradScaler`에 맡깁니다. `fm_loss`, `total_loss`, trainable parameter non-finite 검사는 그대로 유지됩니다.

숫자 안정성을 최우선으로 보고 gradient까지 즉시 fail-fast하고 싶으면 아래처럼 FP32로 실행하세요. 대신 V100에서는 더 느리고 메모리를 더 씁니다.

```bash
... --extra-hydra-overrides 'trainer.precision=32-true'
```

## 2. 환경 설치

권장 환경:

- Linux
- NVIDIA GPU
- Python `3.11.9`
- PyTorch `2.4.x`
- `ffmpeg`

예시:

```bash
conda create -n catk python=3.11.9 -y
conda activate catk

python -m pip install --upgrade pip
python -m pip install -r install/requirements.txt
python -m pip install torch_geometric
python -m pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```

`ffmpeg`는 visualization용으로 필요합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Waymo/WOSAC 자동 제출을 서버에서 사용할 경우, headless browser 런타임 라이브러리도 필요합니다.
최소한 Ubuntu 기준으로는 아래 패키지를 권장합니다.

```bash
apt-get update
apt-get install -y \
  libnss3 \
  libnspr4 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libcups2 \
  libdrm2 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxrandr2 \
  libgbm1 \
  libasound2
```

루트 권한이 없으면 conda env 안에서 아래를 먼저 설치해도 됩니다.

```bash
conda install -y -c conda-forge nss nspr
```

W&B를 쓸 경우:

```bash
wandb login
export WANDB_PROJECT=SMART-FLOW
export WANDB_ENTITY=<your_entity>
```

### 2.1 2025 scorer 관련 주의사항

이 저장소는 시작 시점에 아래를 바로 확인합니다.

- Waymo 공식 2025 Sim Agents scorer를 실제로 불러올 수 있는지
- `traffic_light_violation_likelihood`, `simulated_traffic_light_violation_rate` 같은 2025 전용 필드가 실제 protobuf에 있는지

즉, 예전 Waymo 패키지를 설치하면 validation 시작 전에 명확하게 실패합니다.  
README 기준으로는 `waymo-open-dataset-tf-2-12-0==1.6.7` 이상을 써야 합니다.

## 3. WOMD 데이터 다운로드

이 경로는 **WOMD scenario TFRecord**를 기준으로 합니다.

원하는 위치에 아래 구조가 되도록 준비합니다.

```text
$RAW_ROOT/
├── training/
├── validation/
└── testing/
```

예시 경로:

```bash
export RAW_ROOT=/workspace/womd_v1_3/scenario
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
export CACHE_ROOT=/mnt/nuplan/womd_v1_3/SMART_cache
```

토큰 파일은 저장소에 이미 포함되어 있으므로 별도 다운로드가 필요 없습니다.

- `src/smart/tokens/map_traj_token5.pkl`
- `src/smart/tokens/agent_vocab_555_s2.pkl`

## 4. 캐시 생성

학습과 평가는 원본 TFRecord가 아니라 시나리오별 `.pkl` 캐시를 사용합니다.  
canonical 경로는 `src.data_preprocess`를 직접 호출하는 것입니다.

### 4.1 training 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split training \
  --num_workers 56
```

### 4.2 validation 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split validation \
  --num_workers 56
```

### 4.3 testing 캐시

```bash
python -m src.data_preprocess \
  --input_dir "$RAW_ROOT" \
  --output_dir "$CACHE_ROOT" \
  --split testing \
  --num_workers 56
```

캐시가 끝나면 대략 아래처럼 생깁니다.

```text
$CACHE_ROOT/
├── training/
├── validation/
├── testing/
└── validation_tfrecords_splitted/
```

설명:

- `training/`, `validation/`, `testing/`에는 시나리오별 `.pkl`이 저장됩니다.
- `validation_tfrecords_splitted/`는 `validation` 캐시 생성 시 자동 생성됩니다.
- `validation_tfrecords_splitted/`는 local evaluation, 2025 Sim Agents metric 계산, mp4 visualization에 필요합니다.

### 4.4 Nubes 에서 캐시 다운로드

이미 만들어진 pkl 캐시를 쓰고 싶다면 `scripts/download_smart_cache_from_nubes.sh` 를 사용할 수 있습니다.

기본 사용법:

```bash
bash scripts/download_smart_cache_from_nubes.sh <remote_dir> <local_dir>
```

예시:

```bash
bash scripts/download_smart_cache_from_nubes.sh \
  labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
  "$CACHE_ROOT"
```

또는 환경변수로 넘겨도 됩니다.

```bash
REMOTE_DIR=labs-mlops/ad/research/pnc/hsb/dataset/womd_v1_3/SMART_cache \
LOCAL_DIR="$CACHE_ROOT" \
bash scripts/download_smart_cache_from_nubes.sh
```

## 5. 6x H100에서 Flow Matching 학습

이 경로의 기본 학습 설정은 `configs/experiment/pre_bc_flow.yaml`입니다.

H100 6장 기준 권장 실행:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1006
```

### 5.1 학습 설정을 거칠게 이해하는 법

- 기본 진입점은 `configs/run.yaml`이고, 여기서 `data/model/callbacks/logger/trainer/paths/hydra`를 조합합니다.
- `experiment=pre_bc_flow`는 `configs/experiment/pre_bc_flow.yaml`을 읽어 학습용 하이퍼파라미터를 덮어씁니다.
- `trainer=ddp`는 `configs/trainer/ddp.yaml`을 읽어 DDP 관련 옵션을 덮어씁니다.
- `task_name=...`는 실험 이름이자 저장 폴더 이름입니다. 결과는 대략 `logs/<task_name>/runs/<timestamp>/` 아래에 생깁니다.
- CLI override가 가장 우선입니다. 즉, 같은 파라미터라도 커맨드에 직접 적은 값이 최종 적용됩니다.

예시:

```bash
torchrun ... -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  task_name=flow_semi_continuous_pretrain_h1006
```

### 5.1.1 학습 agent 선택을 validation/추론과 같게 맞추기

기본값은 `data.train_use_eval_agent_selection=false` 입니다.

- `false`면 기존과 같습니다. 학습 입력 agent는 ego 기준 150m 안만 남기고, 학습 대상은 ego/예측 특별 대상과 ego 기준 100m 안이면서 미래 유효 길이가 충분한 agent 중 최대 `data.train_max_num`개를 사용합니다.
- `true`면 학습에서도 validation/추론용 transform을 그대로 사용합니다. 따라서 별도의 150m 입력 제한과 `train_mask` / `train_max_num` 제한을 추가하지 않습니다. 이 경우 학습 입력 agent와 학습 대상 anchor가 validation/추론과 같은 기준으로 정해집니다.
- 이 설정은 pretrain, Flow Matching range fine-tuning, DRaFT fine-tuning에 동일하게 적용됩니다.

예시:

```bash
# pretrain에서 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true
```

### 5.2 Validation 주기와 val_open / val_closed 바꾸기

- 학습 중 validation은 `trainer.check_val_every_n_epoch` 마다 실행됩니다.
- `model.model_config.val_open_loop=true/false`로 open-loop validation on/off를 바꿉니다.
- `model.model_config.val_closed_loop=true/false`로 closed-loop validation on/off를 바꿉니다.
- validation 양 자체는 `trainer.limit_val_batches`로 줄이거나 늘릴 수 있습니다.
- `model.model_config.n_rollout_closed_val`는 `val_closed_loop`에서 scene당 몇 번 rollout sampling할지 정합니다. 현재 `pre_bc_flow` 기본값은 `32`입니다.
- `model.model_config.decoder.closed_loop_rollout_mode=raw_fm|matched_token_chunk`로 closed-loop에서 실제로 export/score/video에 쓰는 10Hz rollout 표현을 고릅니다. 기본값은 `raw_fm`이며, `matched_token_chunk`도 내부 문맥 상태 자체는 실제 FM commit을 유지합니다.
- `model.model_config.decoder.use_stop_motion=true/false`로 stop-motion gate를 켜거나 끕니다.
- `model.model_config.decoder.use_lqr=true/false`로 vehicle / bicycle용 curvature-LQR commit
  bridge를 켜거나 끕니다. 기본값은 `false` 입니다.
- `use_lqr=true`면 2초 미래를 바로 commit하지 않고, 다음 0.5초 commit window만 실제로 실행합니다.
- `use_stop_motion=true`면 stop token 과 일치하는 agent 의 다음 0.5초 5점을 현재 상태로 완전 고정합니다.
- `use_lqr=true`는 stop gate를 통과한 vehicle / bicycle 에만 적용됩니다. pedestrian 은 항상
  token / raw branch 를 유지합니다.
- `model.model_config.n_batch_sim_agents_metric`는 validation 중 공식 2025 scorer를 실제로 돌릴 앞쪽 batch 수입니다. `smart_flow` 기본값은 `10`, `local_val_flow`는 `100`, `sim_agents_sub_flow`는 `0`입니다.
- `trainer.limit_val_batches`는 validation에 실제로 사용할 batch 양입니다. `0.1`이면 전체 validation batch의 10%, `1.0`이면 전체, 정수 `20`이면 앞 20 batch만 평가합니다.
- `data.val_batch_size`는 validation batch당 scene 수입니다. 키우면 validation은 빨라질 수 있지만 GPU memory 사용량도 같이 늘어납니다.
- 공식 2025 scorer 기준 총 채점 scene 수는 대략 `min(실행한 val batch 수, n_batch_sim_agents_metric) x val_batch_size` 입니다.
- closed-loop rollout 총 수는 대략 `(실행한 val batch 수) x val_batch_size x n_rollout_closed_val` 입니다.

예시:

```bash
# 매 epoch마다 validation
... trainer.check_val_every_n_epoch=1

# 5 epoch마다 validation
... trainer.check_val_every_n_epoch=5

# val_open만 실행
... model.model_config.val_open_loop=true model.model_config.val_closed_loop=false

# val_closed만 실행
... model.model_config.val_open_loop=false model.model_config.val_closed_loop=true

# val_closed에서 scene당 rollout 64회
... model.model_config.n_rollout_closed_val=64

# matched token chunk를 실제 closed-loop rollout/video/score 출력에만 사용
... model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

# stop-motion gate 적용
... model.model_config.decoder.use_stop_motion=true

# stop-motion + vehicle / bicycle curvature-LQR commit bridge 적용
... model.model_config.decoder.use_stop_motion=true \
    model.model_config.decoder.use_lqr=true

# use_lqr + matched token chunk를 함께 쓸 때도
# vehicle / bicycle export는 실행된 5점 chunk를 유지하고 pedestrian만 token chunk를 씁니다.
... model.model_config.decoder.use_lqr=true \
    model.model_config.decoder.closed_loop_rollout_mode=matched_token_chunk

# training validation에서 공식 2025 scorer를 앞 20 batch에만 적용
... model.model_config.n_batch_sim_agents_metric=20

# validation을 전체 val set에 대해 수행
... trainer.limit_val_batches=1.0

# validation batch size를 4 -> 2로 줄이기
... data.val_batch_size=2
```

### 5.3 Checkpoint 저장 규칙 바꾸기

- monitored checkpoint 저장 시도는 validation이 도는 시점에 함께 일어납니다. 현재 `pre_bc_flow`는 `check_val_every_n_epoch=8` 이라 기본적으로 8 epoch마다 평가됩니다.
- 현재 기본 기준은 `callbacks.model_checkpoint.monitor=val_closed/sim_agents_2025/realism_meta_metric`, `mode=max`, `save_top_k=1` 입니다. 즉, `realism_meta_metric`이 가장 높은 checkpoint 1개를 유지합니다.
- 저장 위치는 `callbacks.model_checkpoint.dirpath=${paths.output_dir}/checkpoints` 이고, 실제 경로는 `logs/<task_name>/runs/<timestamp>/checkpoints/` 입니다.
- 파일명 규칙은 `callbacks.model_checkpoint.filename="epoch_{epoch:03d}"` 이라 `epoch_002.ckpt` 같은 이름이 됩니다.
- `save_last=link` 이라 `last.ckpt`도 함께 생기며, 저장된 checkpoint를 가리키는 링크로 유지됩니다.
- 별도로 `callbacks.epoch_last_checkpoint.filename=epoch_last.ckpt` 가 매 train epoch의 마지막 batch 직후 현재 상태를 같은 파일에 덮어써 저장합니다. validation이 있는 epoch에서는 validation 시작 전에 먼저 저장되고, validation이 없는 epoch에서도 최신 epoch 기준 checkpoint 1개를 유지합니다.
- validation 중간에 코드가 죽었으면 같은 `epoch_last.ckpt` 로 재개할 때 해당 epoch의 train loop를 다시 돌지 않고, 완료하지 못한 fit-time validation부터 다시 시작하도록 상태를 함께 기록합니다.
- 기본 `logger=wandb` 설정은 `logger.wandb.log_model=all` 이라 저장되는 checkpoint를 W&B model artifact로도 함께 올립니다. 단, `logger.wandb.offline=True` 이거나 `WANDB_MODE=offline|dryrun|disabled` 면 업로드는 자동으로 꺼지고 로컬 checkpoint만 남습니다.
- `epoch_last.ckpt` 는 별도 W&B artifact(`epoch-last-<run_id>`)로도 업로드되며, alias는 항상 `latest`, `epoch_last` 로 갱신됩니다.

자주 바꾸는 파라미터:

- `callbacks.model_checkpoint.monitor`: 어떤 metric으로 best를 고를지
- `callbacks.model_checkpoint.mode=min|max`: metric이 작을수록 좋은지, 클수록 좋은지
- `callbacks.model_checkpoint.save_top_k`: best checkpoint를 몇 개 남길지
- `callbacks.model_checkpoint.filename`: 저장 파일명 패턴
- `callbacks.model_checkpoint.dirpath`: 저장 폴더
- `callbacks.model_checkpoint.save_last=true|link|false`: `last.ckpt`를 어떻게 둘지

예시:

```bash
# val_open/ADE2s가 가장 낮은 checkpoint 3개 저장
... callbacks.model_checkpoint.monitor=val_open/ADE2s \
    callbacks.model_checkpoint.mode=min \
    callbacks.model_checkpoint.save_top_k=3

# checkpoint 파일명을 바꾸기
... callbacks.model_checkpoint.filename='epoch_{epoch:03d}_step_{step}'
```

### 5.4 중단된 학습 재개하기

- 학습 재개 여부는 `task_name`이 아니라 `ckpt_path`로 결정됩니다. 같은 설정으로 다시 실행하면서 이전 run의 checkpoint만 넘기면 됩니다.
- 이 레포는 `trainer.fit(..., ckpt_path=...)`로 재개하므로 model weight뿐 아니라 optimizer, lr scheduler, epoch, global step도 함께 이어집니다.
- monitored checkpoint 기준으로 재개하려면 `logs/<task_name>/runs/<timestamp>/checkpoints/last.ckpt` 가 가장 단순합니다.
- 정확히 가장 최근 train epoch 상태에서 재개하려면 `logs/<task_name>/runs/<timestamp>/checkpoints/epoch_last.ckpt` 를 쓰면 됩니다.
- 현재 `pre_bc_flow` 기본값은 validation이 `8` epoch마다 돌아 monitored checkpoint는 그 시점에만 갱신되지만, `epoch_last.ckpt` 는 매 epoch train loop가 끝나는 즉시 먼저 갱신됩니다.
- validation 도중 크래시가 난 경우에는 `epoch_last.ckpt` 를 다시 넘기면 그 epoch의 validation부터 먼저 다시 시작한 뒤 다음 epoch 학습으로 넘어갑니다.

예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1006 \
  ckpt_path=/path/to/previous_run/checkpoints/last.ckpt
```

다른 PC에서 재개할 때는 그 PC에서 접근 가능한 checkpoint 경로를 `ckpt_path`로 주고, 그 PC의 캐시 위치에 맞게 `paths.cache_root`만 맞춰주면 됩니다. 새로 실행한 쪽의 output dir은 항상 새 timestamp 폴더로 생기므로 기존 run 폴더를 덮어쓰지 않습니다.

### 5.5 `val_closed_loop` 비디오 저장하기

- `pre_bc_flow` 기본값은 `n_vis_batch=0`, `n_vis_scenario=0`, `n_vis_rollout=0` 이라서 `val_closed_loop`가 돌아도 mp4는 저장하지 않습니다.
- 전제: `model.model_config.val_closed_loop=true`

꼭 필요한 파라미터는 아래와 같습니다.

- `model.model_config.n_vis_batch`: validation에서 비디오를 남길 앞쪽 batch 수. 보통 `1~2`부터 시작합니다.
- `model.model_config.n_vis_scenario`: 각 batch에서 저장할 scenario 수. 보통 `1~2`부터 시작하고, 현재 batch 크기 이하로 두면 됩니다.
- `model.model_config.n_vis_rollout`: 각 scenario에서 저장할 rollout 영상 수. 보통 `1~2`부터 시작하고, `n_rollout_closed_val` 이하로 두면 됩니다.
- `model.model_config.vis_ghost_gt=true|false`: rollout 비디오에서 미래 GT agent를 연한 ghost overlay로 같이 그릴지 정합니다. `false`면 `rollout_XX.mp4`에서는 이 연한 GT overlay를 숨기고 sampled rollout만 보입니다. `gt.mp4` 자체는 그대로 저장됩니다.
- `model.model_config.vis_flow_2s_preview=true|false`: rollout 비디오에서 각 0.5초 closed-loop step마다 네트워크가 raw로 생성한 2초 / 20점 future를 overlay로 그릴지 정합니다. `true`면 `rollout_XX.mp4`에서 현재 decision block에 해당하는 raw 20점 궤적이 함께 보입니다.
- `model.model_config.delete_local_videos_after_wandb_upload=true|false`: `wandb`에 비디오를 넘긴 뒤 `logs/.../videos/` 아래 원본 mp4를 지울지 결정합니다. `wandb` logger를 쓰지 않으면 지우지 않습니다.
- 저장 위치는 `logs/<task_name>/runs/<timestamp>/videos/batch_XX-scenario_YY/` 이고, 각 폴더 아래에 `gt.mp4`, `rollout_00.mp4`, `rollout_01.mp4`, ... 형태로 생깁니다. `gt.mp4`는 GT, `rollout_XX.mp4`는 sampled closed-loop rollout입니다. 단, `delete_local_videos_after_wandb_upload=true`면 upload 직후 이 원본 mp4는 자동 삭제될 수 있습니다.
- `logger=wandb` 상태면 생성된 mp4가 W&B에도 같이 기록됩니다. `logger.wandb.offline=True`면 먼저 로컬 `wandb/`에 저장되고, 이후 `wandb sync`로 올리면 됩니다.

예시:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_semi_continuous_pretrain_h1006 \
  model.model_config.n_vis_batch=1 \
  model.model_config.n_vis_scenario=2 \
  model.model_config.n_vis_rollout=2 \
  model.model_config.vis_ghost_gt=false \
  model.model_config.vis_flow_2s_preview=true \
  model.model_config.delete_local_videos_after_wandb_upload=true
```

메모리가 부족하면 아래처럼 train batch를 줄이면 됩니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=pre_bc_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  task_name=flow_pretrain_bs8 \
  data.train_batch_size=8
```

학습 중 W&B에는 기본적으로 아래 metric이 기록됩니다.

- `train/loss`
- `train/ADE2s`
- `train/FDE2s`
- `train/ADEyaw2s`
- `train/FDEyaw2s`
- `val_open/ADE2s`
- `val_open/FDE2s`
- `val_closed/sim_agents_2025/*`
- `val_closed/sim_agents_2025_mean/*`
- `val_closed/sim_agents_2025/minADE_best_of_<n_rollout_closed_val>`

추가로 CUDA OOM 위험도 확인용으로 아래 memory metric이 기록됩니다.

- `worst_peak_reserved_pct`: train batch 1개 기준의 실시간 지표입니다. 각 rank가 자기 GPU의 peak reserved memory 비율(%)을 계산한 뒤, rank 간 `max`로 합친 값입니다. 즉, "그 step에서 가장 위험했던 GPU"를 보여줍니다. W&B에는 20 step 간격으로 샘플링되어 기록됩니다.
- `worst_peak_reserved_pct_epoch_max`: 한 epoch 동안 관측된 `worst_peak_reserved_pct`들 중 최대값입니다. OOM 위험 판단은 이 값을 가장 우선해서 보면 됩니다.

해석 기준은 우선 `worst_peak_reserved_pct_epoch_max`에 적용해서 보면 됩니다. 학습 중 실시간 추세를 볼 때는 `worst_peak_reserved_pct`를 같은 기준으로 봐도 되지만, 최종 판단은 `epoch_max` 기준으로 하는 편이 안전합니다.

- `85%` 미만: 대체로 안정적
- `85% ~ 92%`: 여유가 줄어드는 구간
- `92% ~ 96%`: OOM 고위험 구간
- `97%` 이상: batch 구성이나 입력 길이 스파이크에 따라 바로 OOM이 날 수 있음

추가로 epoch마다 아래 W&B 그래프도 갱신됩니다.

- `training_progress_vs_runtime`: x축은 지금까지 누적된 실제 학습 실행 시간(hours), y축은 전체 epoch 기준 진행률(%)입니다. checkpoint로 학습을 이어서 재개한 경우 이전 runtime도 누적해서 그립니다.

### 5.6 6x H100에서 Flow Matching 학습 범위를 넓혀 fine-tuning

`configs/experiment/finetune_flow_range.yaml`은
**기존 flow checkpoint를 pure Flow Matching loss로 이어서, 학습 범위만 넓혀 새 fine-tuning run을 시작하는 설정**입니다.

핵심은 `data.train_use_eval_agent_selection=true` 입니다.
이 값이 켜지면 학습에서도 validation/추론과 같은 transform을 그대로 써서
기존 학습 경로의 150m 입력 제한과 `train_mask` / `train_max_num` 제한 없이
더 넓은 agent/anchor 범위로 FM loss를 다시 학습합니다.

가장 단순한 6 GPU 실행 예시는 아래와 같습니다.

```bash
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=finetune_flow_range \
  action=finetune \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_range_finetune_h1006
```

중요한 차이:

- 이 경로는 `experiment=pre_bc_flow` + `data.train_use_eval_agent_selection=true`를 매번 길게 적지 않도록 묶어둔 preset입니다.
- `model.model_config.draft.enabled=false` 상태라서 DRaFT inverse feasibility regularizer는 전혀 쓰지 않습니다.
- 즉, **pure FM fine-tuning** 입니다.
- 첫 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`만 읽고 새 optimizer / lr scheduler / epoch / global step으로 다시 시작합니다.
- 따라서 pretrained checkpoint에서 새 FM fine-tuning run을 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기` 방식대로 `action=fit` + 이 fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 써야 합니다.
- `data.train_use_eval_agent_selection=true`일 때는 `WaymoTargetBuilderVal()`을 학습 transform으로 쓰므로 `data.train_max_num`은 실제로 사용되지 않습니다.

`finetune_flow_range` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `16`
- train batch size: `20`
- val batch size: `16`
- validation 주기: `4` epoch마다
- `data.train_use_eval_agent_selection=true`

메모리 관련 주의:

- 이 fine-tuning은 기존 pretrain보다 한 batch 안에 들어오는 agent 수와 학습 대상 anchor 수가 늘 수 있으므로 GPU memory 사용량이 더 커질 수 있습니다.
- 그래서 6x H100 기본 train batch size를 `26 -> 20`으로 낮춰 둔 preset입니다.
- 그래도 OOM이 나면 가장 먼저 `data.train_batch_size`를 `16`, `12`처럼 더 줄이는 편이 안전합니다.

자주 바꾸는 override 예시는 아래와 같습니다.

```bash
# 메모리가 빠듯하면 batch를 더 줄이기
... data.train_batch_size=16

# fine-tuning learning rate를 더 낮추기
... model.model_config.lr=1e-4

# validation을 매 epoch마다 수행
... trainer.check_val_every_n_epoch=1

# 전체 validation set으로 보기
... trainer.limit_val_batches=1.0
```

학습 범위를 "validation/추론과 완전히 같은 기준"으로 넓히는 것이 아니라,
기존 train 규칙 안에서 학습 대상 수만 늘리고 싶다면 아래처럼 하면 됩니다.

```bash
... data.train_use_eval_agent_selection=false data.train_max_num=48
```

다만 이 경우에도 150m 입력 제한과 ego 기준 100m 학습 대상 제한은 그대로 남습니다.

### 5.7 6x H100에서 DRaFT fine-tuning

`configs/experiment/finetune_draft_flow.yaml`을 써서
**기존 flow checkpoint 위에 DRaFT inverse feasibility regularizer를 얹는 fine-tuning**을 바로 시작할 수 있습니다.
이 경로는 pretrain을 이어서 resume하는 용도가 아니라,
**이미 학습된 checkpoint의 weight만 읽어서 새 fine-tuning run을 시작하는 용도**입니다.

가장 단순한 6 GPU 실행 예시는 아래와 같습니다.

```bash
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=finetune_draft_flow \
  action=finetune \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_semi_continuous_finetune_h1006
```

중요한 차이:

- 첫 fine-tuning 시작은 반드시 `action=finetune`를 사용합니다.
- 현재 구현은 `torch.load(ckpt)["state_dict"]`를 `strict=False`로 읽은 뒤 `trainer.fit(...)`을 새로 시작합니다. 단, 현재 fine-tuning에서 `requires_grad=True` 인 파라미터가 checkpoint에 없으면 실행을 중단합니다.
- 즉, optimizer / lr scheduler / epoch / global step은 이어받지 않습니다.
- 반대로 `action=fit`에 `ckpt_path=...`를 주면 **resume training**으로 동작합니다. 이 경우 이전 run의 optimizer 상태까지 이어받습니다.
- 따라서 pretrained checkpoint에서 fine-tuning을 처음 시작할 때만 `action=finetune`를 쓰고,
- 시작한 fine-tuning run이 중단됐으면 그 다음부터는 위 `5.4 중단된 학습 재개하기`
- 방식대로 `action=fit` + fine-tuning run의 `last.ckpt` 또는 `epoch_last.ckpt`를 쓰면 됩니다.

fine-tuning에서 실제로 trainable인 모듈은 아래와 같습니다.

- 기본적으로 encoder 전체를 먼저 freeze합니다.
- `finetune_draft_flow` preset은 `train_full_flow_decoder_only=true`라서
- `agent_encoder.flow_decoder` 전체를 다시 unfreeze합니다.
- 즉 fine-tuning에서는 map encoder, agent embedding, attention layers는 그대로 frozen 상태를 유지하고,
- flow decoder 전체만 trainable 상태로 둡니다.

`finetune_draft_flow` 기본 설정은 아래와 같습니다.

- learning rate: `2e-4`
- max epochs: `32`
- train batch size: `48` per GPU
- effective global train batch size: `288` with 6 GPUs
- val batch size: `16`
- validation 주기: `16` epoch마다
- DRaFT inverse feasibility loss 계산: `model.model_config.draft.loss_enabled=true`

loss와 로그는 아래처럼 보면 됩니다.

- `train/loss`는 최종 학습 loss입니다.
- `train/loss_fm`는 원래 flow matching loss입니다.
- `train/loss_phys`와 `train/loss_if`는 같은 값이고, 새 inverse feasibility penalty `L_if`를 뜻합니다.
- `model.model_config.draft.loss_enabled=true`일 때 실제 학습식은 `train/loss = train/loss_fm + train/draft_weight * 0.005 * train/loss_if` 입니다.
- `model.model_config.draft.loss_enabled=false`로 두면 `draft.max_weight` 값과 무관하게 DRaFT 샘플링과 inverse feasibility loss 계산을 하지 않습니다.
- 이 경우 fine-tuning은 pure Flow Matching으로만 진행되고, `train/loss = train/loss_fm`이 됩니다.
- `loss_enabled=true`인 경우 `train/draft_weight`는 `start_epoch` 이후 `ramp_epochs` 동안 선형으로 증가해 `max_weight`까지 올라갑니다.
- 현재 설정은 `max_weight=0.1`이고, 실제 scale `0.005`는 코드에 고정으로 들어갑니다.
- 따라서 기본 설정의 physics loss 최대 가중치는 `0.1 * 0.005 = 0.0005`입니다.
- 기본 구현은 trainer가 `bf16-mixed`여도 inverse feasibility 계산 구간만 fp32 subregion에서 수행합니다.
- DRaFT physics sample은 FM anchor loss용 train-mode forward를 재사용하지 않고, 생성 모델을 eval mode로 잠깐 바꾼 상태에서 gradient를 유지한 채 다시 만듭니다.
- 따라서 dropout과 history drop이 섞인 학습용 trajectory가 아니라 validation/test와 같은 deterministic inference trajectory를 physics loss로 보정합니다.
- 차량 / 자전거는 예측 20개 점을 다시
- `forward speed`, `curvature`, `steering angle`, `steering rate`, `forward acceleration`으로 바꿔 penalty를 계산합니다.
- wheelbase는 agent box length에 각각 `0.60`, `0.85`를 곱해서 만듭니다.
- 사람은 steering state를 두지 않고, 2차원 속도와 2차원 가속도만으로 hard / soft 항을 계산합니다.
- heading은 속도가 `0.5 m/s`보다 클 때만 약하게 봅니다.
- 첫 제어량은 모두 `prev_control`을 사용합니다.
- 차량 / 자전거는 `v_pre`와 `delta_pre`를 복원해서 첫 가속도와 첫 steering rate를 만들고,
- 사람은 `prev_control[..., :2]`를 `prev_control[..., 2]`의 yaw-rate로 현재 anchor-local 좌표계에 회전한 뒤 첫 2차원 가속도 계산에 씁니다.
- hard 항은 속도, 가속도, steering angle, steering rate, lateral acceleration 제한을 넘는 만큼 `relu(z)^2`로 계산합니다.
- soft 항은 jerk에 가까운 거칠기 값입니다. 기본값에서는 **GT roughness보다 큰 만큼만** loss에 반영하고,
  `model.model_config.draft.physics.compare_softness_to_gt=false` 로 두면
  GT 비교 없이 prediction roughness 자체를 그대로 반영합니다.
- 그래서 `train/loss_phys_raw`와 `train/loss_if_raw`는 GT 비교 전의 raw prediction 기준 값입니다.
- 최종 `L_if`는 agent 전체 평균이 아니라, **batch 안에 실제로 존재하는 class별 평균을 먼저 구한 뒤 다시 class 평균**을 내는 방식입니다.
- 그래서 vehicle이 많아도 pedestrian / bicycle 항이 묻히지 않습니다.
- class별 세부 loss는 `draft_component/*`에 기록됩니다.
- 현재는 `vehicle_hard`, `vehicle_soft`, `vehicle_total`, `bicycle_*`, `pedestrian_hard`, `pedestrian_soft`, `pedestrian_head`, `pedestrian_total`을 봐두면 됩니다.
- 실제 단위 평균값은 `draft_actual_pred/*`, GT 기준값은 `draft_actual_gt/*`에 기록됩니다.
- 현재는 `speed_excess_mps`, `accel_excess_mps2`, `steer_excess_deg`, `steer_rate_excess_degps`, `lat_accel_excess_mps2`, `heading_error_deg`를 남깁니다.

현재 inverse feasibility 기본 하이퍼파라미터는 아래와 같습니다.

- 공통: `soft_weight=0.25`
- vehicle: `v_max=35.0`, `a_max=8.0`, `a_lat_max=4.2`, `wheelbase_scale=0.60`, `steer_max=0.55 rad`, `steer_rate_max=0.8 rad/s`
- bicycle: `v_max=22.0`, `a_max=5.5`, `a_lat_max=4.4`, `wheelbase_scale=0.85`, `steer_max=0.90 rad`, `steer_rate_max=1.4 rad/s`
- pedestrian: `v_max=5.0`, `a_max=4.7`, `heading_speed_threshold=0.5 m/s`, `heading_weight=0.05`

자주 바꾸는 override 예시는 아래와 같습니다.

```bash
# fine-tuning에서도 validation/추론과 같은 agent 기준 사용
... data.train_use_eval_agent_selection=true

# gamma_draft를 더 빨리/강하게 올리기
... model.model_config.draft.max_weight=1.0     model.model_config.draft.ramp_epochs=2

# DRaFT preset의 trainable module / schedule은 쓰되 inverse feasibility loss만 완전히 끄기
... model.model_config.draft.loss_enabled=false

# inverse feasibility도 mixed precision으로 그대로 계산
... model.model_config.draft.physics.force_fp32=false

# soft roughness를 GT와 비교하지 않고 raw prediction 기준으로 사용
... model.model_config.draft.physics.compare_softness_to_gt=false

# 차량 steering rate 제한을 더 느슨하게
... model.model_config.draft.physics.vehicle_steer_rate_max_radps=1.0

# 사람 heading 항을 더 약하게
... model.model_config.draft.physics.pedestrian_heading_weight=0.02

# 샘플러 역전파를 마지막 2 step에만 남겨 메모리 사용량 줄이기
... model.model_config.draft.sampling.backprop_last_k=2

# validation을 매 epoch마다 수행
... trainer.check_val_every_n_epoch=1
```

checkpoint 선택은 보통 아래처럼 하면 됩니다.

- pretrain run의 best 성능 checkpoint를 쓰려면 `epoch_XXX.ckpt`
- 가장 마지막 저장 상태를 쓰려면 `last.ckpt`
- validation 직전까지 포함한 가장 최근 train epoch 상태를 쓰려면 `epoch_last.ckpt`

### 5.8 4x A100 80GB 에서 DRaFT fine-tuning

6x H100 이 아닌 **4x A100 80GB (SXM4)** 박스에서 같은 DRaFT fine-tuning 을 돌리고 싶을 때 쓰는 별도 preset 입니다.

- preset 파일: `configs/experiment/finetune_draft_flow_a100x4.yaml`
- 자세한 실행 방법 / 하이퍼파라미터 선택 이유 / OOM 디버깅 순서: [`docs/A100x4_finetune_draft_flow_README.md`](docs/A100x4_finetune_draft_flow_README.md)

요약만 보면 아래와 같습니다.

- `train_batch_size=36` (실측 max), `accumulate_grad_batches=2`, `trainer.devices=4` → effective global batch **`288`** (6xH100 preset `288` 과 정확히 동일, 따라서 lr 도 그대로 `2e-4`).
- `max_epochs(=32)`, `check_val_every_n_epoch(=16)` 은 6xH100 preset 과 동일.
- `val_batch_size=8` 로 줄이고 `n_rollout_closed_val=16` / `n_batch_sim_agents_metric=10` 은 유지해서 정기 eval 이 OOM 없이 돕니다.
- **bs 상한의 원인은 메모리가 아닙니다**. A100 (sm_80) 의 flash / memory-efficient SDPA kernel 이 `ChunkStepRefiner` 의 self-attention 에서 큰 batch 일 때 `invalid configuration argument` 로 터지는 kernel grid-dim 한계입니다. bs=36 일 때 peak 48 GiB / 80 GiB 로 VRAM 은 남아돕니다.
- 위 crash 를 완전히 없애기 위해 **`src/smart/modules/flow_local_decoder.py` 의 `ChunkStepRefiner` self-attention 만 math-SDPA kernel 로 강제하는 소폭 패치**를 포함했습니다. 실측 결과 bs=36 에서 500 step 이상 안정 + step time 도 오히려 약 20% 단축. 상세: [`docs/A100x4_finetune_draft_flow_README.md`](docs/A100x4_finetune_draft_flow_README.md) 5장.
- 실행 예시:

```bash
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache
export PRETRAIN_CKPT=/path/to/pretrained_flow.ckpt

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m src.run \
  experiment=finetune_draft_flow_a100x4 \
  action=finetune \
  trainer=ddp \
  trainer.devices=4 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$PRETRAIN_CKPT" \
  task_name=flow_semi_continuous_finetune_inv_best_a_100_a100x4
```

#### 5.8.1 No-DRaFT ablation + adaptive train_batch_size sweep

같은 4x A100 박스에서 **DRaFT 를 통째로 끄고** (`model.model_config.draft.enabled=false`, sampling/loss 모두 비활성), `train_batch_size` 를 64 부터 시작해 OOM 이 나면 4 씩 줄여 재시도하는 스크립트입니다. ablation 비교군 (DRaFT 적용 vs 미적용) 을 자동화하기 위한 러너입니다.

- 스크립트: `scripts/finetune_a100x4_no_draft_bs_sweep.sh`
- 첫 시도: `action=finetune` (pretrained weight 로 epoch 0 시작)
- OOM 발생 시: `action=fit ckpt_path=<직전 attempt 의 epoch_last.ckpt>` 로 Lightning full-resume — **마지막으로 완료된 epoch 부터 이어서** 학습 (epoch counter / optimizer / scheduler 모두 복원). 한 epoch 도 끝나기 전에 OOM 이 나면 더 작은 bs 로 pretrain 부터 다시 시작.
- OOM 이 아닌 에러 (모델 버그 / I/O 등) 면 즉시 abort — bs 줄여도 안 풀리는 문제이므로.
- bs sweep: 64 → 60 → 56 → ... → 4
- `accumulate_grad_batches=1` 로 고정. 따라서 effective global batch = `bs × 4`.
- **Linear LR scaling** 자동 적용: `lr = 2e-4 × (bs × 4) / 288` (기준점은 원본 preset 의 `bs=36, accum=2` 글로벌 288 / lr 2e-4). 단 `action=fit` 으로 resume 할 때는 Lightning 이 ckpt 의 optimizer state 를 복원하므로 lr override 가 무시됩니다 (의도된 보수 동작).

실행:

```bash
cd /mnt/nuplan/projects/catk
bash scripts/finetune_a100x4_no_draft_bs_sweep.sh
```

각 attempt 의 raw torchrun 출력은 `/tmp/${TASK_NAME}_attempt<N>_bs<bs>.log` 에 저장돼서 OOM 판정 / 디버깅에 쓰입니다.

### 5.8.2 Feasible DRaFT slip-angle penalty

이 변경은 DRaFT fine-tuning의 물리 feasibility 항에 vehicle / bicycle 전용 slip-angle penalty를 추가합니다. 목적은 도로 정합성, 주변 agent 상호작용, RMM metric 직접 최적화가 아니라, heading 방향과 실제 이동 방향이 크게 어긋나는 비물리적인 옆방향 미끄러짐을 줄이는 것입니다.

각 0.1초 구간에서 현재 anchor 상태를 `(0, 0, 0)`으로 두고, 미래 위치 변화량을 직전 heading 기준 body 좌표계로 회전합니다.

```text
vx_body = (dx * cos(theta_prev) + dy * sin(theta_prev)) / dt
vy_body = (-dx * sin(theta_prev) + dy * cos(theta_prev)) / dt
beta = atan2(abs(vy_body), abs(vx_body) + eps)
```

vehicle / bicycle에만 아래 제한값을 적용합니다.

```text
vehicle beta_max = 0.27 rad
bicycle beta_max = 0.70 rad
```

초과량은 `semi_continuous_lqr`의 proxy penalty와 같은 dead-zone 제곱 형태를 사용합니다.

```text
r = relu(beta - beta_max) / (abs(beta_max) + eps)
z = (r - 0.02) / 0.02
slip_penalty = (0.02 * softplus(z))^2
```

최종 vehicle / bicycle feasibility 손실은 기존 hard 항과 기존 soft 처리 방식 사이에 slip 항을 1.0 배율로 직접 더합니다.

```text
vehicle_or_bicycle_total = hard + slip + soft_weight * soft_effective
```

기존 sampling 설정, draft 전체 가중치, train batch size, learning rate는 바꾸지 않습니다. 학습 로그에는 `vehicle_slip`, `bicycle_slip`, `slip_beta_excess_deg`가 추가됩니다.

## 6. 평가와 추론

### 6.1 Validation set closed-loop 평가

`configs/experiment/local_val_flow.yaml`은 validation split에서 closed-loop rollout을 수행하고, Waymo 공식 2025 Sim Agents metric을 계산합니다.  
가장 단순한 사용법은 single GPU 평가입니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=local_val_flow \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_local_val
```

이 명령은 아래를 한 번에 수행합니다.

- validation split inference
- closed-loop rollout
- `val_closed/sim_agents_2025/*`
- `val_closed/sim_agents_2025_mean/*`
- `val_closed/sim_agents_2025/minADE_best_of_32`

주의:

- `local_val_flow` 기본값은 `trainer.limit_val_batches=60` 이라 빠른 local check용입니다.
- 전체 validation set을 돌리고 싶으면 `trainer.limit_val_batches=1.0` 을 추가하면 됩니다.
- 현재 `local_val_flow`는 `model.model_config.n_batch_sim_agents_metric=100` 이라 실행한 validation batch 전체에 대해 공식 scorer를 돌립니다.

### 6.2 Validation set에서 open-loop만 보고 싶을 때

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=local_val_flow \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_open_val \
  model.model_config.val_open_loop=true \
  model.model_config.val_closed_loop=false
```

### 6.3 6 GPU로 validation inference를 병렬화하고 싶을 때

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=local_val_flow \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_local_val_ddp
```

## 7. WOSAC 2025 제출 파일 생성

`configs/experiment/sim_agents_sub_flow.yaml`은 **Waymo/WOSAC에 올릴 제출 파일을 만드는 설정**입니다.
점수를 계산하는 설정이 아니라, 최종 제출용 `tar.gz`를 만드는 설정이라고 생각하면 됩니다.

헷갈리기 쉬운 차이는 아래처럼 보면 됩니다.

- `local_val_flow`: validation 점수를 보고 싶을 때
- `sim_agents_sub_flow`: 제출 파일을 만들고 싶을 때
- `action=validate`: validation split으로 제출 형식이 잘 나오는지 미리 확인할 때
- `action=test`: test split으로 최종 제출 파일을 만들 때

`sim_agents_sub_flow`는 기본적으로 아래처럼 동작합니다.

- 제출 파일 생성 모드로 실행됩니다.
- 로컬 점수는 계산하지 않습니다.
- validation/test split 전체를 읽도록 기본값이 잡혀 있습니다.

실행 전에 아래 값은 꼭 채워 주세요.

- `ckpt_path`
- `model.model_config.sim_agents_submission.method_name`
- `model.model_config.sim_agents_submission.authors`
- `model.model_config.sim_agents_submission.affiliation`
- `submission.description` 또는 `model.model_config.sim_agents_submission.description`
- `model.model_config.sim_agents_submission.method_link`
- `model.model_config.sim_agents_submission.account_name`

`ckpt_path`에는 보통 아래 중 하나를 넣으면 됩니다.

- 가장 최근 학습 상태를 쓰려면 `last.ckpt` 또는 `epoch_last.ckpt`
- 가장 성능이 좋았던 checkpoint를 쓰려면 `epoch_XXX.ckpt`

### 7.1 validation split으로 제출 형식 먼저 확인하기

`action=validate`는 validation 데이터를 읽어서 제출 파일이 잘 만들어지는지 확인하는 용도입니다.
점수를 계산하는 명령은 아니므로, validation 점수도 함께 보고 싶다면 `local_val_flow`를 따로 한 번 더 실행해야 합니다.

빠르게 1 GPU로 형식만 확인하고 싶다면:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_validate \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME"
```

### 7.2 validation split 전체를 6 GPU로 제출 파일 만들기

validation split 전체를 6 GPU로 나눠서 빠르게 처리하고 싶다면 아래 명령을 쓰면 됩니다.
실행이 끝나면 validation 기준 제출 파일 `tar.gz`가 만들어집니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_val_ddp6_step_16 \
  trainer.limit_val_batches=1.0 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME" \
  paths.log_dir=/workspace/exp_logs
```

이 명령에서 중요한 옵션만 보면 아래와 같습니다.

- `action=validate`: validation split을 사용합니다.
- `trainer=ddp`, `trainer.devices=6`: GPU 6장을 함께 사용합니다.
- `trainer.limit_val_batches=1.0`: validation split 전체를 끝까지 읽습니다.
- `model.model_config.val_open_loop=false`: open-loop 계산은 생략합니다.
- `model.model_config.val_closed_loop=true`: 제출 파일 생성에 필요한 closed-loop rollout은 유지합니다.
- `paths.log_dir=/workspace/exp_logs`: 로그를 저장할 위치입니다.

### 7.2.1 validation split 전체를 8 GPU (V100-SXM2 32GB × 8) 로 제출 파일 만들기

7.2 의 H100 80GB × 6 preset 을 Tesla V100-SXM2 **32GB × 8** 박스에 이식한 버전입니다.

핵심 차이:

- **V100 은 bfloat16 하드웨어가 없습니다** (sm_70, Volta). `trainer.precision=bf16-mixed` 대신 **`trainer.precision=16-mixed`** (FP16 mixed precision) 를 씁니다.
- **GPU 8 장**: `nproc_per_node=8`, `trainer.devices=8`.
- **`data.val_batch_size=8`**: V100 × 8 에서 steady-state throughput sweep 으로 확인한 최적값입니다 (실측 표는 아래).

실행 커맨드는 아래와 같습니다.

```bash
export CACHE_ROOT=/workspace/womd_v1_3/SMART_cache   # 각자의 캐시 경로로 교체
export CKPT=/path/to/epoch_last.ckpt                 # 예: wandb artifact 로컬 저장 경로

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=ddp \
  trainer.devices=8 \
  trainer.precision=16-mixed \
  trainer.limit_val_batches=1.0 \
  data.val_batch_size=8 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path="$CKPT" \
  task_name=flow_sim_agents_val_v100x8 \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME" \
  paths.log_dir=/workspace/exp_logs
```

#### val_batch_size 선택 근거 (V100 × 8, 실측)

`scripts/bench/v100x8_sim_agents_sub_val_sweep.sh` 로 측정한 결과. 조건: 8× V100-SXM2 32GB, `precision: 16-mixed`, `n_rollout_closed_val=32`, `num_workers=4`, validation 은 `torch.no_grad()` 라 backward activation 이 없어 peak VRAM 이 매우 작습니다.

| `val_batch_size` | batch ms | total samples/s (8 GPU) | peak VRAM | 32 GiB 마진 | 상태 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 1  | 2,286  | 3.50 | 2.5 GiB  | 29.5 GiB | ✅ |
| 4  | 6,847  | 4.67 | 5.9 GiB  | 26.1 GiB | ✅ |
| **8**  | **12,583** | **5.09** | **11.6 GiB** | **20.4 GiB** | ✅ **채택 (peak throughput)** |
| 16 | 26,995 | 4.74 | 21.2 GiB | 10.8 GiB | 오히려 느려짐 (−7%) |
| 32 | 58,584 | 4.37 | 30.4 GiB | 1.6 GiB (risky) | 더 느려지고 OOM 마진 위험 |
| 64 | — | — | OOM | — | ❌ |

관찰:

- **Throughput 은 `val_bs=8` 에서 peak**. 이후 bs 를 더 키우면 samples/s 가 오히려 감소합니다 (bs=8 → 16 에서 −7%).
- 이유: closed-loop rollout 은 80-step 순차 시뮬레이션이라 GPU kernel 이 bs=8 에서 이미 compute-bound. 더 키우면 scene variance (큰 agent 수의 worst-case step) 가 batch 전체 step 시간을 끌어내립니다.
- 메모리 측면에서는 bs=32 까지도 넣을 수 있지만 throughput 관점에서 의미가 없고, bs=32 는 dense scene 에서 32 GiB 를 넘길 수 있어 **bs=8 이 wall-clock 최소 + 안전 마진 최대** 의 sweet spot 입니다.
- bs=1 은 GPU 가 놀아서 throughput 이 크게 떨어집니다 (3.50 vs 5.09 samples/s, **−31%**).

#### 44,097 샘플 전체 export 예상 시간

| `val_batch_size` | 전체 소요 | 비고 |
|:--:|:--:|:--|
| **8 (채택)** | **~ 2.40 h** | 44,097 / 5.09 samples/s |
| 16 | 2.58 h | +8% |
| 4  | 2.62 h | +9% |
| 32 | 2.80 h | +17%, OOM 마진 위험 |
| 1  | 3.50 h | +46% |

참고로 H100 × 6 preset 은 같은 validation split 에서 BF16 + bs=4 로 약 1.5–2 시간 수준입니다. V100 × 8 은 하드웨어 상 compute / 메모리 대역폭이 H100 보다 좁아 1.3~1.6 배 느린 것이 정상입니다.

#### OOM 이나 다른 문제가 날 때

- 만약 다른 ckpt 로 peak VRAM 이 여기 표 대비 크게 오르면 (예: chunk_mixer hidden 이 넓은 구버전 artifact), `data.val_batch_size: 8 → 4` 로 먼저 낮춰 보세요.
- CPU RAM 이 부족하면 `data.num_workers: 4 → 3`, `data.prefetch_factor: 1` 유지 (기본값).
- `trainer.precision=bf16-mixed` 는 V100 에서 에러가 나거나 FP32 보다 느린 software emulation 으로 돌 수 있으므로 **반드시 `16-mixed`** 를 쓰세요.

#### 재현

스윕 스크립트: `scripts/bench/v100x8_sim_agents_sub_val_sweep.sh`. 사용법:

```bash
CACHE_ROOT=/workspace/womd_v1_3/SMART_cache \
CKPT=/path/to/epoch_last.ckpt \
CANDIDATES="1 4 8 16 32" LIMIT=6 \
bash scripts/bench/v100x8_sim_agents_sub_val_sweep.sh
```

각 후보마다 `[VALBENCH] val_bs=.. batch_ms=.. total_samples_s=.. peak_vram_mib=..` 한 줄이 `scripts/bench/v100x8_sim_agents_sub_results.log` 에 남습니다.

### 7.3 test split으로 최종 제출 파일 만들기

실제로 Waymo/WOSAC에 올릴 test split 결과를 만들 때는 `action=test`를 사용합니다.
validation 예시와 비교하면 핵심 차이는 `action=test` 하나입니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=test \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_test \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  paths.log_dir=/workspace/exp_logs
```

실행이 끝나면 아래 파일이 생성됩니다.

- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission/`
- `logs/<task_name>/runs/<timestamp>/sim_agents_2025_submission.tar.gz`

validation export와 test export는 저장 위치와 파일 형식이 같습니다.
차이는 validation 데이터를 읽었는지, test 데이터를 읽었는지만 다릅니다.

알아둘 점:

- `sim_agents_sub_flow`는 제출 파일 생성용이라 로컬 점수는 계산하지 않습니다.
- 점수와 제출 파일이 둘 다 필요하면 `local_val_flow`와 `sim_agents_sub_flow`를 각각 한 번씩 실행해야 합니다.
- 특별한 이유가 없으면 `n_rollout_closed_val=32`는 그대로 두는 편이 안전합니다.
- 메모리가 부족하면 `data.val_batch_size` 또는 `data.test_batch_size`를 `4 -> 2 -> 1` 순서로 줄여 보세요.
- validation split export는 형식 확인용으로 좋고, 실제 업로드는 보통 test split에서 만든 `tar.gz`를 사용합니다.

### 7.4 SSH 서버에서 Waymo 사이트로 자동 업로드

SSH 서버에서도 제출 파일을 만든 뒤 바로 Waymo 사이트에 업로드할 수 있습니다.
다만 Google 로그인은 한 번 필요하므로, **GUI가 있는 PC에서 로그인 상태를 저장한 뒤**
서버에서는 그 JSON 내용을 그대로 붙여넣는 방식으로 쓰는 편이 가장 안전합니다.
같은 파일을 서버 저장소에 오래 남겨 둘 필요는 없습니다.

로그인 상태 파일의 기본 위치는 아래와 같습니다.

```text
secrets/waymo/waymo_storage_state.json
```

이 파일은 로그인된 상태를 그대로 담고 있으므로 비밀번호처럼 조심해서 다뤄야 합니다.
공개 저장소에는 올리지 않는 편이 안전합니다.
현재 `.gitignore`에는 `secrets/waymo/waymo_storage_state.json` 과
`secrets/waymo/playwright_profile/` 이 포함되어 있습니다.

준비:

```bash
python -m pip install -r install/requirements.txt
python -m playwright install chromium
```

환경에 `python` 명령이 없으면 아래 예시의 `python`을 전부 `python3`로 바꿔서 실행하면 됩니다.

1. GUI가 있는 PC에서 로그인 상태를 저장합니다.

```bash
python scripts/waymo_save_storage_state.py --browser-channel chrome
```

기본 저장 위치는 `secrets/waymo/waymo_storage_state.json` 입니다.  
로그인이 잘 안 되면 Playwright 기본 Chromium보다 설치된 Chrome이나 Edge를 쓰는 편이 더 안정적입니다.
그래서 GUI PC에서는 `--browser-channel chrome` 또는 `--browser-channel msedge`를 권장합니다.
이 스크립트는 저장 직전에 Sim Agents 페이지를 다시 확인해서 실제 업로드 폼이 보이는지 검증합니다.
즉, Google 로그인만 된 상태가 아니라 **`Submit to Validation Set` / `Submit to Test Set` 업로드 박스가 실제로 보여야** 저장이 완료됩니다.
Waymo가 `Review rules`를 보여주면 그 자리에서 약관 동의를 한 번 마친 뒤 다시 저장해야 합니다.
스크립트는 SSH/headless 업로드에 필요한 `waymo.com`의 localStorage
(`datasetChallengeTermsAgreementAccepted=true`)도 함께 `waymo_storage_state.json`에 넣어 둡니다.

추가로 기억할 점:

- 브라우저 프로필은 실행할 때마다 임시로 만들고, 종료하면 정리합니다.
- `--user-data-dir`를 직접 줄 때는 Playwright 전용의 빈 폴더를 쓰는 편이 안전합니다.
- 평소 쓰는 기본 Chrome 프로필 폴더를 그대로 넣는 건 권장하지 않습니다.
- 예전에 만든 프로필을 재사용하다가 브라우저가 바로 꺼지면 `--user-data-dir` 없이 다시 실행해 보세요.
- 서버에 이 파일을 꼭 복사해 둘 필요는 없습니다. 아래 자동 업로드 명령을 실행하면,
  서버에 파일이 없을 때 rank 0 프로세스가 시작 직후 터미널에 JSON 붙여넣기를 요청합니다.
- 서버에도 파일을 두고 싶다면 `waymo_submission.storage_state_path` 경로에 배치하면 되고,
  그 경우에는 붙여넣기 프롬프트 없이 기존 파일을 그대로 사용합니다.

2. 서버에서 자동 업로드를 켠 상태로 validation 또는 test를 실행합니다.

validation 예시는 아래와 같습니다.  
서버에 `secrets/waymo/waymo_storage_state.json` 파일이 없으면, 이 명령은 시작 직후
rank 0에서 로컬 파일 내용 전체를 붙여넣으라고 묻습니다. pretty-printed JSON을 그대로 붙여넣고
마지막 `}` 뒤에서 Enter를 한 번 더 치면 검증이 바로 이어집니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  -m src.run \
  experiment=sim_agents_sub_flow \
  action=validate \
  trainer=ddp \
  trainer.devices=6 \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_sim_agents_val_ddp6_step_16 \
  trainer.limit_val_batches=1.0 \
  model.model_config.val_open_loop=false \
  model.model_config.val_closed_loop=true \
  model.model_config.sim_agents_submission.method_name="SMART-flow-7M" \
  model.model_config.sim_agents_submission.authors=[Anonymous] \
  model.model_config.sim_agents_submission.affiliation="YOUR_AFFILIATION" \
  model.model_config.sim_agents_submission.description="YOUR_DESCRIPTION" \
  model.model_config.sim_agents_submission.method_link="YOUR_METHOD_LINK" \
  model.model_config.sim_agents_submission.account_name="YOUR_ACCOUNT_NAME" \
  waymo_submission.enabled=true \
  waymo_submission.poll_submission_status=false \
  paths.log_dir=/workspace/exp_logs
```

이때 입력된 JSON은 `/tmp` 아래의 임시 파일로만 저장되고, 프로세스 종료 시 자동으로 삭제됩니다.
즉, 서버 저장소 안에 `waymo_storage_state.json`을 따로 커밋하거나 유지하지 않아도 됩니다.

핵심 옵션은 아래만 기억하면 됩니다.

- `waymo_submission.enabled=true`: 자동 업로드를 켭니다.
- `waymo_submission.storage_state_path`: 로그인 상태 파일 경로입니다. 기본값은 `secrets/waymo/waymo_storage_state.json` 입니다.
  이 파일이 서버에 있으면 그대로 쓰고, 없으면 실행 시작 시 JSON 붙여넣기를 요청합니다.
- `waymo_submission.poll_submission_status=false`: 업로드 후 점수 페이지를 계속 확인하지는 않습니다.

추가 참고:

- validation 실행에서는 `waymo_submission.enabled=true`만 주면 업로드까지 진행됩니다.
- `torchrun` DDP에서도 rank 0만 한 번 입력을 받고, 나머지 rank는 그 입력이 끝날 때까지 대기합니다.
- 서버에서 기본으로 headless Chromium을 사용합니다.
- 서버에 설치된 Chrome을 쓰고 싶으면 `waymo_submission.browser_channel=chrome` 또는 `waymo_submission.browser_executable_path=/path/to/chrome`를 지정하면 됩니다.
- 현재 코드는 Chromium launch 전에 `CONDA_PREFIX/lib`를 자동으로 `LD_LIBRARY_PATH` 앞에 추가하고,
  Playwright bundled browser 외에도 system Chrome과 `~/.cache/ms-playwright/chromium-*/chrome-linux/chrome`
  경로를 자동 탐색해 순서대로 재시도합니다.
- 브라우저가 서버 라이브러리 부족 등으로 launch에 실패하면, 현재 코드는 저장된 `waymo_storage_state.json` 쿠키를 사용해 Waymo 업로드 API로 자동 fallback 합니다.
- 저장한 상태 파일이 불완전하면 업로드 단계에서 `Review rules` 또는 로그인 게이트가 잡히도록 에러 메시지가 분명하게 나옵니다.
  이 경우에는 GUI PC에서 `python scripts/waymo_save_storage_state.py --browser-channel chrome`를 다시 실행하고,
  Sim Agents 페이지에 실제 업로드 폼이 보이는 상태에서 저장한 파일로 교체하면 됩니다.
- 로그인 만료나 페이지 구조 변경으로 실패하면 `logs/<task_name>/runs/<timestamp>/waymo_submission_debug/` 아래에 디버그 파일이 남습니다.
- 점수 페이지까지 자동 확인하고 싶으면 `waymo_submission.poll_submission_status=true`를 줄 수 있지만, UI 변경에 영향을 받을 수 있어 기본값은 `false`입니다.

test 자동 제출은 실수 방지를 위해 기본으로 꺼져 있습니다.
Waymo test set은 계정당 30일에 3번만 제출할 수 있으므로, test 업로드를 할 때는 아래 옵션을 추가로 넣어야 합니다.

```bash
... action=test \
    waymo_submission.enabled=true \
    waymo_submission.submit_test=true
```

즉, `waymo_submission.enabled=true`만으로는 test 제출이 올라가지 않습니다.

## 8. Visualization

학습 중 `val_closed_loop` 비디오 저장 방법은 위 `5.5 val_closed_loop 비디오 저장하기`를 참고하면 됩니다.  
checkpoint로 validation visualization만 따로 보고 싶으면 아래처럼 `local_val_flow`를 쓰면 됩니다.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m src.run \
  experiment=local_val_flow \
  trainer=default \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  paths.cache_root="$CACHE_ROOT" \
  ckpt_path=/path/to/model.ckpt \
  task_name=flow_local_val_vis \
  model.model_config.n_vis_batch=2 \
  model.model_config.n_vis_scenario=5 \
  model.model_config.n_vis_rollout=5 \
  model.model_config.vis_ghost_gt=false \
  model.model_config.vis_flow_2s_preview=true \
  model.model_config.delete_local_videos_after_wandb_upload=true
```

비디오 저장 위치:

```text
logs/<task_name>/runs/<timestamp>/videos/
```

생성되는 파일:

- `gt.mp4`
- `rollout_00.mp4`
- `rollout_01.mp4`
- ...

W&B logger를 켜 둔 경우 같은 mp4가 W&B에도 함께 업로드됩니다.

## 9. 빠른 체크리스트

학습 전:

- `training/` 캐시 존재
- `validation/` 캐시 존재
- `validation_tfrecords_splitted/` 존재
- `paths.cache_root="$CACHE_ROOT"` 확인
- Waymo 2025 scorer 환경 확인

WOSAC 2025 test submission 전:

- `testing/` 캐시 존재
- `ckpt_path` 확인
- submission metadata 6개 필드 확인
- `experiment=sim_agents_sub_flow` 확인

WOSAC 2025 validation submission export 전:

- `validation/` 캐시 존재
- `validation_tfrecords_splitted/` 존재
- `ckpt_path` 확인
- submission metadata 6개 필드 확인
- `experiment=sim_agents_sub_flow action=validate` 확인
- `trainer.limit_val_batches=1.0` 확인

## 10. 자주 쓰는 명령 모음

### 캐시 생성

```bash
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split training --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split validation --num_workers 56
python -m src.data_preprocess --input_dir "$RAW_ROOT" --output_dir "$CACHE_ROOT" --split testing --num_workers 56
```

### 6x H100 학습

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=pre_bc_flow trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" task_name=flow_semi_continuous_pretrain_h1006
```

### validation 평가

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.run experiment=local_val_flow trainer=default trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_local_val
```

### test submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=sim_agents_sub_flow action=test trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_test
```

### validation submission export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 -m src.run experiment=sim_agents_sub_flow action=validate trainer=ddp trainer.devices=6 paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_val_ddp6 trainer.limit_val_batches=1.0 model.model_config.val_open_loop=false model.model_config.val_closed_loop=true
```

### validation submission export (V100 × 8)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" torchrun --standalone --nproc_per_node=8 -m src.run experiment=sim_agents_sub_flow action=validate trainer=ddp trainer.devices=8 trainer.precision=16-mixed trainer.limit_val_batches=1.0 data.val_batch_size=8 model.model_config.val_open_loop=false model.model_config.val_closed_loop=true paths.cache_root="$CACHE_ROOT" ckpt_path=/path/to/model.ckpt task_name=flow_sim_agents_val_v100x8
```
