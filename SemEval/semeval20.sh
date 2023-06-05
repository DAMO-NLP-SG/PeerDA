#!/bin/bash
hostname
nvidia-smi
task=SemEval20
CUDA_VISIBLE_DEVICES=0 python $task/train-SemEval20.py \
        --output_dir ./saved_models/DA-roberta-base-semeval20-2e-5 \
        --model_type roberta \
        --model_name_or_path roberta-base --cache_dir ../cache \
        --data_path ./Data/semeval20 \
        --do_train --do_eval \
        --learning_rate 2e-5 \
        --num_train_epochs 20 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=32 \
        --max_seq_length 200 --max_query_length 80 \
        --save_steps 0 --logging_steps 5000 \
        --fp16 --expand_rate 0.5 --gradient_accumulation_steps 1  \
        --overwrite_output_dir --evaluate_during_training --DA