#!/bin/bash

pid_datasets="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"

for pid_dataset in ${pid_datasets}
do
    python \
    train.py \
    --model_fn model.pth \
    --model_name bert4kt_rasch \
    --dataset_name ${pid_dataset} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done