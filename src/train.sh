#!/bin/bash
model_names="monaconvbert4kt_plus convbert4kt_plus"
dataset_names="assist2009_pid assist2012_pid assist2017_pid algebra2005_pid algebra2006_pid"

for dataset_name1 in ${dataset_names1}
do
    python \
    train.py \
    --model_fn reestimate2_monoconvbert4kt_rmse.pth \
    --model_name monoconvbert4kt_plus \
    --dataset_name ${dataset_name1} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --crit rmse \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

for dataset_name2 in ${dataset_names2}
do
    python \
    train.py \
    --model_fn reestimate2_monoconvbert4kt_rmse.pth \
    --model_name monoconvbert4kt_plus \
    --dataset_name ${dataset_name2} \
    --num_encoder 12 \
    --batch_size 128 \
    --grad_acc True \
    --grad_acc_iter 4 \
    --crit rmse \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done

for dataset_name3 in ${dataset_names3}
do
        python \
        train.py \
        --model_fn reestimate2_convbert4kt_plus_auc.pth \
        --model_name convbert4kt_plus \
        --dataset_name ${dataset_name3} \
        --num_encoder 12 \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --use_leakyrelu True \
        --fivefold True \
        --n_epochs 1000
done

for dataset_name4 in ${dataset_names4}
do
    python \
    train.py \
    --model_fn reestimate2_convbert4kt_rmse.pth \
    --model_name convbert4kt_plus \
    --dataset_name ${dataset_name4} \
    --num_encoder 12 \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --crit rmse \
    --use_leakyrelu True \
    --fivefold True \
    --n_epochs 1000
done