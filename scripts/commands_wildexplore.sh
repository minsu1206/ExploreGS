#!/bin/bash
# diffusion / system / gs / uncertainty / virtualcam
# example : 
# bash scripts/run_stage2_360.sh bicycle 0 ver1-3_resize_minib4 debug_train_evalr hard_occlusion fisher_test_exceptrot spherev3_dbg 30000;
# bash scripts/sweep_nerfbusters_stage2.sh 0 ver1-3_resize_minib4 debug_diffcam ddistort1 fisher_test_exceptrot sphere-ablation-radius4 30000
# bash scripts/run_stage2.sh table 0 enhancer3 default default visibility_mask search_v1_simple_dbg2 default 30000 local
# bash scripts/run_stage_vcam.sh aloe 0 enhancer3 default default visibility_mask search_free_nbvs_lookcenter_dbg depthreg 30000 remote
chmod -R 777 output
gpu=1
exp=depthreg
bash scripts/sweep_wildexplore_stage1.sh ${gpu} ${exp} local
exp=ours
bash scripts/sweep_wildexplore_stage2.sh ${gpu} enhancer default full visibility_mask ${exp} depthreg 30000 local ${exp}