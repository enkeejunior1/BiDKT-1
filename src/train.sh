#!/bin/bash
dataset_names="assist2009_pid_diff_pt"

for dataset_name in ${dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_diff_rmse.pth \
    --model_name monaconvbert4kt_plus_diff \
    --dataset_name ${dataset_name} \
    --num_encoder 12 \
    --batch_size 64 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --use_leakyrelu True \
    --crit rmse \
    --fivefold True \
    --n_epochs 1000
done

for algebra_dataset in ${algebra_dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_diff_rmse.pth \
    --model_name monaconvbert4kt_plus_diff \
    --dataset_name ${algebra_dataset} \
    --num_encoder 12 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --use_leakyrelu True \
    --crit rmse \
    --fivefold True \
    --n_epochs 1000
done

