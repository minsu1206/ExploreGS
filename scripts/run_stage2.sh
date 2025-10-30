#!/bin/bash
scene=$1
gpu=$2
cd=$3
cs=$4
cg=$5
cu=$6
cv=$7
ckpt=$8
ckpt_iter=$9
dataset=${10}
logname=${11}_${scene}
swap=${12:-0}
expname=${13:-${11}}  # If $8 is empty, set expname to an empty string

echo "swap: $swap"
echo "expname: $expname"

cfg_r=configs/stage2
port=$((6006 + gpu))
ip="127.0.0.$((6 + gpu))"
ckpt_dir=output/stage1_${ckpt}/${scene}
if [ $dataset == "local" ]; then
    dataset_dir=/workspace/dataset/nerfbusters-dataset/${scene}
    # ckpt_dir=output/stage1_${ckpt}/curated_nb/${scene}
    # ckpt_dir=output/stage1_${ckpt}/${scene}
    # original: output/stage1_${ckpt}/nerfbusters/${scene}
elif [ $dataset == "remote" ]; then
    dataset_dir=/workspace/dataset_lustre/users/minsu/dataset/nerfbusters-dataset/${scene}
    # ckpt_dir=/workspace/dataset_lustre/users/minsu/ckpt/stage1_${ckpt}/curated_nb/${scene}
    # original: ckpt_dir=/workspace/dataset_lustre/users/minsu/ckpt/stage1_${ckpt}/nerfbusters/${scene}
else
    echo "Invalid dataset option"
    exit 1
fi

if [ $(($swap)) -eq 1 ]; then
    ckpt_dir=${ckpt_dir}_swap
fi

# Build the command with conditional inclusion of --expname
cmd="CUDA_VISIBLE_DEVICES=$gpu python train_stage_vcam.py \
    --logname ${logname} \
    -s ${dataset_dir} \
    --eval \
    -r 1 \
    -cd ${cfg_r}/diffusion/${cd} \
    -cs ${cfg_r}/system/${cs} \
    -cg ${cfg_r}/gs/${cg} \
    -cu ${cfg_r}/uncertainty/${cu} \
    -cv ${cfg_r}/virtualcam/${cv} \
    --load Nerfbusters \
    --start_checkpoint ${ckpt_dir}/chkpnt${ckpt_iter}.pth \
    --port ${port} \
    --ip ${ip}"

# Add --expname only if expname is not empty
if [[ -n "$expname" ]]; then
    cmd+=" --expname ${expname}"
fi

if [ $(($swap)) -eq 1 ]; then
    cmd+=" --swap"
fi

# Execute the command
eval $cmd
# exit 0

cmd="CUDA_VISIBLE_DEVICES=$gpu python train_stage_diffusion.py \
    --logname ${logname} \
    -s ${dataset_dir} \
    --eval \
    -r 1 \
    -cd ${cfg_r}/diffusion/${cd} \
    -cs ${cfg_r}/system/${cs} \
    -cg ${cfg_r}/gs/${cg} \
    -cu ${cfg_r}/uncertainty/${cu} \
    -cv ${cfg_r}/virtualcam/${cv} \
    --load Nerfbusters \
    --start_checkpoint ${ckpt_dir}/chkpnt${ckpt_iter}.pth \
    --port ${port} \
    --ip ${ip}"

# Add --expname only if expname is not empty
if [[ -n "$expname" ]]; then
    cmd+=" --expname ${expname}"
fi

if [ $(($swap)) -eq 1 ]; then
    cmd+=" --swap"
fi

eval $cmd

cmd="CUDA_VISIBLE_DEVICES=$gpu python train_stage_finetune.py \
    --logname ${logname} \
    -s ${dataset_dir} \
    --eval \
    -r 1 \
    -cd ${cfg_r}/diffusion/${cd} \
    -cs ${cfg_r}/system/${cs} \
    -cg ${cfg_r}/gs/${cg} \
    -cu ${cfg_r}/uncertainty/${cu} \
    -cv ${cfg_r}/virtualcam/${cv} \
    --load Nerfbusters \
    --start_checkpoint ${ckpt_dir}/chkpnt${ckpt_iter}.pth \
    --checkpoint_iterations 15000 30000 \
    --port ${port} \
    --ip ${ip}"

# Add --expname only if expname is not empty
if [[ -n "$expname" ]]; then
    cmd+=" --expname ${expname}"
fi

if [ $(($swap)) -eq 1 ]; then
    cmd+=" --swap"
fi

eval $cmd

cmd="CUDA_VISIBLE_DEVICES=$gpu python render.py --logname ${logname} \
    -s ${dataset_dir} \
    -cd ${cfg_r}/diffusion/${cd} \
    -cs ${cfg_r}/system/${cs} \
    -cg ${cfg_r}/gs/${cg} \
    -cu ${cfg_r}/uncertainty/${cu} \
    -cv ${cfg_r}/virtualcam/${cv} \
    --start_checkpoint ${ckpt_dir}/chkpnt${ckpt_iter}.pth \
    --load Nerfbusters"

if [[ -n "$expname" ]]; then
    cmd+=" --expname ${expname}"
fi

if [ $(($swap)) -eq 1 ]; then
    cmd="$cmd --swap"
fi

echo $cmd
eval $cmd

cmd="CUDA_VISIBLE_DEVICES=$gpu python metrics.py --logname ${logname} \
    -s ${dataset_dir} \
    -cd ${cfg_r}/diffusion/${cd} \
    -cs ${cfg_r}/system/${cs} \
    -cg ${cfg_r}/gs/${cg} \
    -cu ${cfg_r}/uncertainty/${cu} \
    -cv ${cfg_r}/virtualcam/${cv} \
    --start_checkpoint ${ckpt_dir}/chkpnt${ckpt_iter}.pth"

if [[ -n "$expname" ]]; then
    cmd+=" --expname ${expname}"
fi

if [ $(($swap)) -eq 1 ]; then
    cmd="$cmd --swap"
fi

eval $cmd
