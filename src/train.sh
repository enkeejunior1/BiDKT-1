#!/bin/bash
dataset_names="assist2009_pid_diff assist2012_pid_diff assist2017_pid_diff"
algebra_dataset_names="algebra2005_pid_diff algebra2006_pid_diff"

for dataset_name in ${dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_diff.pth \
    --model_name monaconvbert4kt_plus_diff \
    --dataset_name ${dataset_name} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

for algebra_dataset in ${algebra_dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_diff.pth \
    --model_name monaconvbert4kt_plus_diff \
    --dataset_name ${algebra_dataset} \
    --num_encoder 12 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

