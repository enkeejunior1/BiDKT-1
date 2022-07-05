#!/bin/bash

pid_datasets="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"
num_encoders="6 8 10 12"

for pid_dataset in ${pid_datasets}
do
    for num_encoder in ${num_encoders}
    do
        python \
        train.py \
        --model_fn model.pth \
        --model_name bert4kt_plus \
        --dataset_name ${pid_dataset} \
        --num_encoder ${num_encoder} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done