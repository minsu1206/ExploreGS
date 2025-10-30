#!/bin/bash
gpu=$1
cd=$2
cs=$3
cg=$4
cu=$5
cv=$6
ckpt=$7
ckpt_iter=$8
dataset=$9
logname=${10}
expname=${11}

scenes=(aloe art century flowers garbage picnic roses)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage2.sh ${scene} ${gpu} ${cd} ${cs} ${cg} ${cu} ${cv} ${ckpt} ${ckpt_iter} ${dataset} ${logname} 1 ${expname}
done

scenes=(pipe table)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage2.sh ${scene} ${gpu} ${cd} ${cs} ${cg} ${cu} ${cv} ${ckpt} ${ckpt_iter} ${dataset} ${logname} ${expname}
done

echo "Sweep nerfbusters done!"
