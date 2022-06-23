#!/bin/bash

datasets="assist2015 assist2009"
num_encoders="12 24"
grad_acc_iters="8"

for dataset in ${datasets}
do
    for num_encoder in ${num_encoders}
    do
        python \
        train.py \
        --model_fn model.pth \
        --dataset_name ${dataset} \
        --num_encoder ${num_encoder} \
        --grad_acc True \
        --grad_acc_iter ${grad_acc_iter} \
        --fivefold True \
        --n_epochs 500
    done
done