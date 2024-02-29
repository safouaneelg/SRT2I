#! /usr/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="../optimal_pairs"

#   original --max_train_steps : 15000, changed to 500 for test
accelerate launch ../txt2im/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=500 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd_optim_pairs_test"