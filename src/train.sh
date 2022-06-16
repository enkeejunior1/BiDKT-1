#!/bin/bash

list=""
model_name="bidkt"

for num in ${list}
do
    python \
    train.py \
    --model_fn model.pth \
    --dataset_name coldstart1 \
    --model_name ${model} \
    --five_fold True \
    --record_path ../records/coldstart1_record.tsv \
    --n_epochs 100
done