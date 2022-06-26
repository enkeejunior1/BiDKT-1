#!/bin/bash

datasets="assist2015 assist2009 algebra2005 algebra2006"
crits="binary_cross_entropy rmse"

for crit in ${crits}
do
    for dataset in ${datasets}
    do
        python \
        train.py \
        --model_fn model.pth \
        --dataset_name ${dataset} \
        --num_encoder 24 \
        --crit ${crit} \
        --batch_size 256 \
        --grad_acc True \
        --grad_acc_iter 2 \
        --fivefold True \
        --n_epochs 1000
    done
done
