#!/bin/bash

datasets="assist2015 assist2009"
batchs="8 16 32 64"
hidden_size="64 128 256 512"

for dataset in ${datasets}
do
    for batch in ${batchs}
    do
        for hidden in ${hidden_size}
        do
            python \
            train.py \
            --model_fn model.pth \
            --batch_size ${batch} \
            --dataset_name ${dataset}
            --n_epochs 100
        done
    done
done