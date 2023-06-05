#!/bin/bash
hostname
nvidia-smi
task=NER
CUDA_VISIBLE_DEVICES=0 python $task/train-NER.py \
        --output_dir ./saved_models/DA-roberta-large-weibo-1e-5 \
        --model_type bert \
        --model_name_or_path hfl/chinese-roberta-wwm-ext-large --cache_dir ../cache \
        --data_path ./Data/weibo \
        --do_train --do_eval \
        --learning_rate 1e-5 \
        --num_train_epochs 20 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=8 \
        --max_seq_length 192 --max_query_length 64\
        --save_steps 0 --logging_steps 2000\
        --fp16 --expand_rate 1 --gradient_accumulation_steps 1\
        --overwrite_output_dir --evaluate_during_training --is_chinese --DA
