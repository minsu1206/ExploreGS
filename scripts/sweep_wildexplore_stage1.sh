#!/bin/bash
gpu=$1
cfg=$2
dataset=$3
scenes=(study hat kitchen towtractor livingroom vacantlot table2 bicycle)
for scene in "${scenes[@]}"; do
    bash scripts/run_stage1_we.sh ${scene} ${gpu} ${cfg} ${dataset}
done
