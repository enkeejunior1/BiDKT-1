#!/bin/bash

datasets2="assist2009 assist2015 assist2017"
encoder_nums="12 16"

for dataset2 in ${datasets2}
do
    for encoder_num in ${encoder_nums}
    do
        python \
        train.py \
        --model_fn model.pth \
        --dataset_name ${dataset2} \
        --num_encoder ${encoder_num} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done

for dataset2 in ${datasets2}
do
    for encoder_num in ${encoder_nums}
    do
        python \
        train.py \
        --model_fn model.pth \
        --dataset_name ${dataset2} \
        --num_encoder ${encoder_num} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu False \
        --fivefold True \
        --n_epochs 1000
    done
done