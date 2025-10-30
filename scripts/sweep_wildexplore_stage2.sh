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

scenes=(study hat kitchen towtractor livingroom vacantlot table2 bicycle)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage2_we.sh ${scene} ${gpu} ${cd} ${cs} ${cg} ${cu} ${cv} ${ckpt} ${ckpt_iter} ${dataset} ${logname} ${expname}
done

echo "Sweep wildexplore done!"  