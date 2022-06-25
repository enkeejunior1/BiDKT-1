#!/bin/bash

datasets="assist2015 assist2009 algebra2005 algebra2006"
num_encoders="12 24"
grad_acc_iter="2"
crits="binary_cross_entropy rmse"

#batch 256
for crit in ${crits}
do
    for dataset in ${datasets}
    do
        for num_encoder in ${num_encoders}
        do
            python \
            train.py \
            --model_fn model.pth \
            --dataset_name ${dataset} \
            --num_encoder ${num_encoder} \
            --crit ${crit} \
            --batch_size 256 \
            --fivefold True \
            --n_epochs 1000
        done
    done
done

#batch 512
for crit in ${crits}
do
    for dataset in ${datasets}
    do
        for num_encoder in ${num_encoders}
        do
            python \
            train.py \
            --model_fn model.pth \
            --dataset_name ${dataset} \
            --num_encoder ${num_encoder} \
            --crit ${crit} \
            --grad_acc True \
            --grad_acc_iter ${grad_acc_iter} \
            --batch_size 256 \
            --fivefold True \
            --n_epochs 1000
        done
    done
done
