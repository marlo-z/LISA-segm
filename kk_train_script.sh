#!/bin/bash

deepspeed --include=localhost:0 --master_port=24999 train_ds.py \
  --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
  --dataset_dir='./dataset' \
  --vision_pretrained="./pretrained_weights/sam_vit_h_4b8939.pth" \
  --dataset="refer_seg" \
  --sample_rates="1" \
  --exp_name="lisa++" \
