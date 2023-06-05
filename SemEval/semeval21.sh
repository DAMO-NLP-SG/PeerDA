#!/bin/bash
hostname
nvidia-smi
task=SemEval21
CUDA_VISIBLE_DEVICES=3 python $task/train-SemEval21.py \
        --output_dir ./saved_models/DA-roberta-base-semeval21-3e-5 \
        --model_type roberta \
        --model_name_or_path roberta-base --cache_dir ../cache \
        --data_path ./Data/semeval21 \
        --do_train --do_eval \
        --learning_rate 3e-5 \
        --num_train_epochs 10 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 200 --max_query_length 80 \
        --save_steps 0 --logging_steps 2000 \
        --fp16 --expand_rate 1 --gradient_accumulation_steps 1  \
        --overwrite_output_dir --evaluate_during_training --DA