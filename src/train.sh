#!/bin/bash

datasets="assist2015 assist2009"
batchs="8 16 32 64"
num_encoders="12 16 24"
hidden_size="64 128 256 512"

for dataset in ${datasets}
do
    for batch in ${batchs}
    do
        for num_encoder in ${num_encoders}
        do
            for hidden in ${hidden_size}
            do
                python \
                train.py \
                --model_fn model.pth \
                --batch_size ${batch} \
                --dataset_name ${dataset} \
                --num_encoder ${num_encoder} \
                --n_epochs 100
            done
        done
    done
done

#grad_accumeration
grad_acc_iters = "2 4 8"

for dataset in ${datasets}
do
    for num_encoder in ${num_encoders}
    do
        for hidden in ${hidden_size}
        do
            for grad_acc_iter in ${grad_acc_iters}
            do
                python \
                train.py \
                --model_fn model.pth \
                --dataset_name ${dataset} \
                --num_encoder ${num_encoder} \
                --grad_acc True \
                --grad_acc_iter ${grad_acc_iter} \
                --n_epochs 100
            done
        done
    done
done