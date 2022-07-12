#!/bin/bash

model_names="monoconvbert4kt_plus"
model_names2="convbert4kt_plus"
dataset_names="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"
dataset_names2="assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"
dataset_names3="algebra2005_pid algebra2006_pid"

# LeakyReLU 측정
for model_name in ${model_names}
do
    for dataset_name3 in ${dataset_names3}
    do
        python \
        train.py \
        --model_fn reestimate2_bert_auc.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset_name3} \
        --num_encoder 12 \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 4 \
        --use_leakyrelu True \
        --fivefold True \
        --batch_size 128 \
        --n_epochs 1000
    done
done

for model_name in ${model_names}
do
    for dataset_name2 in ${dataset_names2}
    do
        python \
        train.py \
        --model_fn reestimate2_bert_rmse.pth \
        --model_name ${model_name} \
        --dataset_name ${dataset_name2} \
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

# LeakyReLU 측정
for model_name2 in ${model_names2}
do
    for dataset_name2 in ${dataset_names2}
    do
        python \
        train.py \
        --model_fn reestimate2_bert_auc.pth \
        --model_name ${model_name2} \
        --dataset_name ${dataset_name2} \
        --num_encoder 12 \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
    done
done

for model_name2 in ${model_names2}
do
    for dataset_name in ${dataset_names}
    do
        python \
        train.py \
        --model_fn reestimate2_bert_rmse.pth \
        --model_name ${model_name2} \
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