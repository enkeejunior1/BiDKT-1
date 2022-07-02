#!/bin/bash
pid_datasets="assist2009_pid assist2017_pid algebra2005_pid algebra2006_pid slepemapy_pid"

for pid_dataset in ${pid_datasets}
do
    python \
    train.py \
    --model_fn model.pth \
    --model_name ma_bert4kt_plus \
    --dataset_name ${pid_dataset} \
    --num_encoder 12 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done