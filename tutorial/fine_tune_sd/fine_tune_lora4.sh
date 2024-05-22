#! /usr/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export TRAIN_DIR="../../optimal_pairs4"
export VAL_PROMPT="A giraffe standing in a field of wildflowers, with a sense of vibrancy and life. Background: wildflower field, clear blue sky, and a warm, sunny day."
export VAL_PROMPT2="Two elephants standing at the edge of a cliff, looking out over a vast expanse of sea. Background: rocky cliffs, crashing waves, and a sense of awe and wonder."

accelerate launch --config_file cfg/config_gpu0.yaml ../../txt2im/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --random_flip \
  --train_batch_size=18 \
  --mixed_precision="fp16" \
  --num_train_epochs=100 --checkpointing_steps=1500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="soptimpair4_lora_100_base_2" \
  --validation_prompt="Two elephants standing at the edge of a cliff, looking out over a vast expanse of sea. Background: rocky cliffs, crashing waves, and a sense of awe and wonder." \
  --validation_epochs=10 \
  --num_validation_images=20 \
  --report_to="wandb"