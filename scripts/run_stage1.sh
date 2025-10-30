#!/bin/bash
scene=$1
gpu=$2
cfg=$3
dataset=$4
swap=${5:-0}
cfg_r=configs/stage1
# if dataset is local : dataset_dir = /workspace/dataset/nerfbusters-dataset/${scene}
# if dataset is remote : dataset_dir = /workspace/dataset_lustre/nerfbusters-dataset/${scene}
if [ $dataset == "local" ]; then
    dataset_dir=/workspace/dataset/nerfbusters-dataset/${scene}
elif [ $dataset == "remote" ]; then
    dataset_dir=/workspace/dataset_lustre/users/minsu/dataset/nerfbusters-dataset/${scene}
else
    echo "Invalid dataset option"
    exit 1
fi
port=$((6006 + gpu))
ip="127.0.0.$((6 + gpu))"

cmd="CUDA_VISIBLE_DEVICES=$gpu python train_stage1.py \
    -s ${dataset_dir} \
    --eval \
    -r 1 \
    -c ${cfg_r}/${cfg} \
    --load Nerfbusters \
    --port ${port} \
    --ip ${ip}"

if [ $swap == 1 ]; then
    cmd="$cmd --swap"
fi

eval $cmd

scene_name=${scene}

cmd="CUDA_VISIBLE_DEVICES=$gpu python render.py -m output/stage1_${cfg}/${scene_name} \
    -c ${cfg_r}/${cfg}.yaml \
    -s ${dataset_dir} --load Nerfbusters --stage1"

if [ $swap == 1 ]; then
    cmd="$cmd --swap"
fi

eval $cmd

cmd="CUDA_VISIBLE_DEVICES=$gpu python metrics_stage1.py -m output/stage1_${cfg}/${scene_name}"
if [ $swap == 1 ]; then
    cmd="$cmd --swap"
fi

eval $cmd
