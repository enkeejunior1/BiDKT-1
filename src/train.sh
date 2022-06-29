#!/bin/bash
encoder_nums="12 16"

for encoder_num in ${encoder_nums}
do
    python \
    train.py \
    --model_fn bert4kt_plus_leakyrelu_model.pth \
    --model_name bert4kt_plus \
    --dataset_name assist2017_pid \
    --num_encoder ${encoder_num} \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

datasets="assist2009 assist2015 assist2017"

#gelu
for dataset in ${datasets}
do
    for encoder_num in ${encoder_nums}
    do
        python \
        train.py \
        --model_fn bidkt_gelu_model.pth \
        --dataset_name ${dataset} \
        --num_encoder ${encoder_num} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu False \
        --fivefold True \
        --n_epochs 1000
    done
done