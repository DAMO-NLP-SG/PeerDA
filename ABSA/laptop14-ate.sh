#!/bin/bash

hostname
nvidia-smi
task=ABSA
for seed in 42 0 1 2 3 4
do


CUDA_VISIBLE_DEVICES=3 python $task/train-ABSA.py \
        --output_dir ./saved_models/DA-bert-large-laptop14-2e-5-ate \
        --model_type bert \
        --model_name_or_path bert-large-uncased --cache_dir ../cache \
        --data_path ./Data/laptop14 \
        --do_train --do_eval --do_lower_case \
        --learning_rate 3e-5 \
        --num_train_epochs 30 \
        --per_gpu_eval_batch_size=32  \
        --per_gpu_train_batch_size=16 \
        --max_seq_length 128 --max_query_length 24\
        --save_steps 0 --logging_steps 2000 \
         --fp16 --expand_rate 1 --expand_method 0 --seed $seed\
        --overwrite_output_dir --evaluate_during_training --DA --ate
done

