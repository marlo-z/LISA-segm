#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0 \

deepspeed --master_port=24999 train_ds.py \
  --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
  --dataset_dir='./dataset' \
  --vision_pretrained="./pretrained_weights/sam_vit_h_4b8939.pth" \
  --dataset="refer_seg" \
  --sample_rates="1" \
  --exp_name="lisa++" \
  --refer_seg_data="refcoco" \
  --box_min_size=400 \
  --val_dataset="refcoco|unc|val" \
