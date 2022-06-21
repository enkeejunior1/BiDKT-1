#!/bin/bash

datasets="assist2015 assist2009"
num_encoders="12 16 24"
batch_size="64"
grad_acc_iters="2 4 8"

for dataset in ${datasets}
do
    for num_encoder in ${num_encoders}
    do
        python \
        train.py \
        --model_fn model.pth \
        --dataset_name ${dataset} \
        --batch_size ${batch_size} \
        --num_encoder ${num_encoder} \
        --fivefold True \
        --n_epochs 100
    done
done

for dataset in ${datasets}
do
    for num_encoder in ${num_encoders}
    do
        for grad_acc_iter in ${grad_acc_iters}
        do
            python \
            train.py \
            --model_fn model.pth \
            --dataset_name ${dataset} \
            --batch_size ${batch_size} \
            --num_encoder ${num_encoder} \
            --grad_acc True \
            --grad_acc_iter ${grad_acc_iter} \
            --fivefold True \
            --n_epochs 100
        done
    done
done