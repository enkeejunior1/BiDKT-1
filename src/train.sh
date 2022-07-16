#!/bin/bash
dataset_names="assist2009_pid_diff_pt"

for dataset_name in ${dataset_names}
do
    python \
    train.py \
    --model_fn monaconvbert4kt_plus_diff_pt.pth \
    --model_name monaconvbert4kt_plus_diff_pt \
    --dataset_name ${dataset_name} \
    --num_encoder 12 \
    --batch_size 64 \
    --grad_acc True \
    --grad_acc_iter 8 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

