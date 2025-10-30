#!/bin/bash
gpu=$1
cfg=$2
dataset=$3

scenes=(aloe art century flowers garbage picnic roses)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage1.sh ${scene} ${gpu} ${cfg} ${dataset} 1
done
scenes=(pipe table)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage1.sh ${scene} ${gpu} ${cfg} ${dataset} 0
done
echo "Sweep nerfbusters done!"