#!/bin/bash
scene=$1
gpu=$2
cfg=$3
dataset=$4
cfg_r=configs/stage1 
# if dataset is local : dataset_dir = /workspace/dataset/nerfbusters-dataset/${scene}
# if dataset is remote : dataset_dir = /workspace/dataset_lustre/nerfbusters-dataset/${scene}
if [ $dataset == "local" ]; then
    dataset_dir=/workspace/dataset/wild-explore/${scene}
elif [ $dataset == "remote" ]; then
    dataset_dir=/workspace/dataset_lustre/users/minsu/dataset/wild-explore/${scene}
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
    --load WildExplore \
    --port ${port} \
    --ip ${ip}"

eval $cmd

scene_name=${scene}

for traj in train interpolation extrapolation explore1 explore2; do
    cmd="CUDA_VISIBLE_DEVICES=$gpu python render_we.py -m output/stage1_${cfg}/${scene_name} \
        -c ${cfg_r}/${cfg}.yaml \
        -s ${dataset_dir} --load WildExplore --render_${traj} --stage1"
    echo $cmd
    eval $cmd
done

for traj in train interpolation extrapolation explore1 explore2; do
    cmd="CUDA_VISIBLE_DEVICES=$gpu python metrics_we.py -m output/stage1_${cfg}/${scene_name} --render_traj ${traj} --stage1"
    eval $cmd
done
