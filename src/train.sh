#!/bin/bash

datasets1="assist2009 assist2017 algebra2005 algebra2006 statics slepemapy"

for dataset1 in ${datasets1}
do
    python \
    train.py \
    --model_fn model.pth \
    --dataset_name ${dataset1} \
    --num_encoder 12 \
    --crit binary_cross_entropy \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --fivefold True \
    --use_gelu True \
    --n_epochs 1000
done

datasets2="assist2015 assist2009 assist2017 algebra2005 algebra2006 statics slepemapy"

for dataset2 in ${datasets2}
do
    python \
    train.py \
    --model_fn model.pth \
    --dataset_name ${dataset2} \
    --num_encoder 12 \
    --crit rmse \
    --batch_size 256 \
    --grad_acc True \
    --grad_acc_iter 2 \
    --fivefold True \
    --use_gelu True \
    --n_epochs 1000
done