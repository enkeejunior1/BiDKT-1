#!/bin/bash

pid_datasets1="assist2012_pid slepemapy_pid"

for pid_dataset1 in ${pid_datasets1}
do
    python \
    train.py \
    --model_fn convbert4kt_plus.pth \
    --model_name convbert4kt_plus \
    --dataset_name ${pid_dataset1} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

pid_datasets2="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid slepemapy_pid"

for pid_dataset2 in ${pid_datasets2}
do
    python \
    train.py \
    --model_fn convbert4kt_plus.pth \
    --model_name convbert4kt_plus \
    --dataset_name ${pid_dataset2} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu False \
    --fivefold True \
    --n_epochs 1000
done