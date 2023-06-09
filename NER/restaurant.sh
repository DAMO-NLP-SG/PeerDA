#!/bin/bash

hostname
nvidia-smi
task=NER

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12359 $task/train-NER.py \
        --output_dir ./saved_models/DA-roberta-base-restaurant-1e-5 \
        --model_type roberta \
        --model_name_or_path roberta-base --cache_dir ../cache \
        --data_path ./Data/restaurant \
        --do_train --do_eval --do_lower_case \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --per_gpu_eval_batch_size=16  \
        --per_gpu_train_batch_size=32 \
        --max_seq_length 160 --max_query_length 64\
        --save_steps 0 --logging_steps 5000\
        --fp16 --expand_rate 1 --gradient_accumulation_steps 1 \
        --overwrite_output_dir --evaluate_during_training --DA