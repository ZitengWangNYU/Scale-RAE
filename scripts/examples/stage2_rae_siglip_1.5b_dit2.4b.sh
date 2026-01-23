#!/bin/bash
#
# Scale-RAE Stage 2: Instruction Tuning
# Fine-tune Stage 1 checkpoint on high-quality instruction data
#

########################################################################################
# Environment Setup
########################################################################################
export XLA_DISABLE_FUNCTIONALIZATION=1
export PJRT_DEVICE=TPU

export EXPERIMENT_NAME="stage2_rae_siglip_1.5b_dit2.4b"
export GCS_BUCKET_NAME="your-gcs-bucket"
export GCS_CHECKPOINT_DIR="scale-rae-checkpoints"
export GCS_OUTPUT_DIR="gs://${GCS_BUCKET_NAME}/${GCS_CHECKPOINT_DIR}/${EXPERIMENT_NAME}"

export WANDB_PROJECT="Scale-RAE"
export WANDB_ENTITY="your-wandb-entity"
export WANDB_NAME="${EXPERIMENT_NAME}"
export WANDB_API_KEY="your-wandb-api-key"

########################################################################################
# Data Configuration
########################################################################################
DATA_PATH="/path/to/instruction_data.jsonl"  # High-quality instruction tuning data
IMAGE_FOLDER=""

########################################################################################
# Model Configuration
########################################################################################
# Load Stage 1 checkpoint
PRETRAINED_CKPT_PATH="your-username/scale-rae-1.5b-dit2.4b-stage1"
# Or from GCS: gs://your-bucket/checkpoints/stage1_rae_siglip_1.5b_dit2.4b

########################################################################################
# Training Configuration
########################################################################################
TRAIN_ARGS="
    --model_name_or_path ${PRETRAINED_CKPT_PATH} \
    --version qwen_2 \
    --vision_loss_mode query \
    --vision_loss diffusion-loss \
    --vision_coef 2.0 \
    --diffusion_split_per_token 256 \
    --diffusion_model_hidden_size 2048 \
    --diffusion_model_z_channels 2048 \
    --diffusion_model_heads 32 \
    --diffusion_model_depth 32 \
    --ddt_encoder_depth 2 \
    --diffusion_class_dropout_prob 0.1 \
    --diff_head_lr 5.66e-4 \
    --diff_head_constant_schedule false \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower_aux_list [\"google/siglip2-so400m-patch14-224\"] \
    --vision_tower_aux_token_len_list [256] \
    --vision_hidden_size 1152 \
    --connector_only True \
    --mm_projector_type mlp2x_gelu \
    --unfreeze_mm_vision_tower False \
    --mm_vision_tower_lr 2e-6 \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 False \
    --output_dir ${GCS_OUTPUT_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 5.66e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 False \
    --max_grad_norm 1.0 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${EXPERIMENT_NAME} \
    --fsdp full_shard \
    --fsdp_config fsdp_config_rf.json \
    --max_images_per_sample 1 \
    --anyres_max_subimages 4 \
    --si_token_len 729 \
    --miv_token_len 0 \
    --video_fps 1 \
    --video_max_frames 4 \
    --video_force_sample True \
    --add_time_instruction True \
    --resume_from_checkpoint True \
"

echo "=========================================="
echo "Scale-RAE Stage 2: Instruction Tuning"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output: ${GCS_OUTPUT_DIR}"
echo "Loading from: ${PRETRAINED_CKPT_PATH}"
echo "=========================================="

python scale_rae/train/train_spmd.py ${TRAIN_ARGS}

echo "Stage 2 instruction tuning completed!"

