#!/bin/bash
encoder_nums="12 16"

pid_datasets="assist2009_pid assist2017_pid algebra2005_pid algebra2006_pid slepemapy_pid"

for pid_dataset in ${pid_datasets}
do
    for encoder_num in ${encoder_nums}
    do
        python \
        train.py \
        --model_fn bert4kt_plus_leakyrelu_model.pth \
        --model_name bert4kt_plus \
        --dataset_name ${pid_dataset} \
        --num_encoder ${encoder_num} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done