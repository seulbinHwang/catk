CONDA_SH=/home2/pnc2/miniforge3/etc/profile.d/conda.sh
[ -f "$CONDA_SH" ] && . "$CONDA_SH"
conda activate catk
cd /home2/pnc2/repos_python/project_basic_adjoint_theory
BS = 24
VAL_BS=$((BS/2))
echo "=== PROBE train_bs=$BS val_bs=$VAL_BS ==="

WANDB_MODE=offline WANDB_SILENT=true \
CUDA_VISIBLE_DEVICES=2,3 \
LOGLEVEL=INFO HYDRA_FULL_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 --master_port=42793 --rdzv_endpoint=127.0.0.1:42793 \
-m src.run \
experiment=am_finetune_flow \
task_name=am_probe_bs${BS} \
ckpt_path=/home2/pnc2/repos_python/project/logs/pretrained/epoch_last.ckpt \
paths.cache_root=/home2/pnc2/repos_python/datasets/catk_cache \
+trainer.max_steps=20 \
trainer.max_epochs=1 \
trainer.num_sanity_val_steps=0 \
trainer.limit_val_batches=0 \
data.train_batch_size=${BS} \
data.val_batch_size=${VAL_BS} \
data.train_max_num=8 \
logger.wandb.entity=se99an

