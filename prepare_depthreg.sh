#!/bin/bash
# hat starbucks2
scene=$1
gpu=$2
root=/workspace/dataset/scannetpp_dslr_only
# CUDA_VISIBLE_DEVICES=${gpu} python run.py --encoder vitl --pred-only --grayscale --img-path ${root}/${scene}/images_2 --outdir ${root}/${scene}/depths
# cd ../
# python utils/make_depth_scale.py --scene ${scene} --dataset wild-explore


# only for scannetpp_dslr_only
# scenes=(0a7cc12c0e 0d2ee665be 1ada7a0617 1f7cbbdde1 3e8bba0176 5eb31827b7 5ee7c22ba0 7b6477cb95 13c3e046d7 21d970d8de 31a2c91c43 38d58a7a31 40aec5fffa)
scenes=(40aec5fffa)
scenes=(45b0dac5e3 99fa5c25e1 825d228aec 927aacd5d1)
cd Depth-Anything-V2
for scene in "${scenes[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu} python run.py --encoder vitl --pred-only --grayscale --img-path ${root}/${scene}/dslr/resized_undistorted_images --outdir ${root}/${scene}/dslr/depths
    cd ../
    python utils/make_depth_scale.py --scene ${scene} --dataset scannetpp_dslr_only
    cd Depth-Anything-V2
done
echo "Sweep nerfbusters done!"
