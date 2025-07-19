#!/bin/bash
#SBATCH -p rtx6000           # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:1    # request GPU(s)
#SBATCH -c 8              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=sahi_inference_2x_zoom_SR_swinir_bioscan_v9_05_025_conf_final_val_set_results_0.6_overlap_0.25_conf_orig_scale
#SBATCH --exclude=gpu177,gpu138,gpu124,gpu121,gpu127
###SBATCH --job-name=sahi_inference_128_sahi_tiled_v9_05_025_conf_final_val_set_results_0.6_overlap_0.25_conf_orig_scale

source ~/.bashrc
source activate mask_rcnn
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

########################## VARIABLE TILE SIZE - 15K ITERS ######################################################
TILE_SIZE=512

python sahi_inference.py --model_path /h/jquinto/Mask-RCNN/output_${TILE_SIZE}_sahi_tiled_v9/model_final.pth \
--exp_name lifeplan_b_${TILE_SIZE}_sahi_tiled_keep_cutoff_v9_R50_one_cycle_5e-5_epoch_3_15k_iters_TEST_SET \
--dataset_json_path /h/jquinto/lifeplan_b_v9_cropped_center/annotations/instances_test2017.json \
--dataset_img_path /h/jquinto/lifeplan_b_v9_cropped_center/test2017 \
--config_path /h/jquinto/Mask-RCNN/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--crop_fac 16 \
--postprocess_match_threshold 0.5 \
--model_confidence_threshold 0.25 \
--predict \
--scale_factor 1 \
--slice_height ${TILE_SIZE} \
--slice_width ${TILE_SIZE} \
--overlap 0.6