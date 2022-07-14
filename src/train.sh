#!/bin/bash

model_names="monaconvbert4kt_plus convbert4kt_plus"
dataset_names="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"

# LeakyReLU 측정
for model_name in ${model_names}
do
    for dataset_name in ${dataset_names}
    do
        python \
        train.py \
        --model_fn reestimate_bert_auc.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset_name} \
        --num_encoder 12 \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done

for model_name in ${model_names}
do
    for dataset_name in ${dataset_names}
    do
        python \
        train.py \
        --model_fn reestimate_bert_rmse.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset_name} \
        --num_encoder 12 \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --crit rmse \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done