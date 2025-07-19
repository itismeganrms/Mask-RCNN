#!/bin/bash
#SBATCH -p rtx6000            # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:4       # request GPU(s)
#SBATCH -c 4              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=mask_rcnn_lifeplan_b_512_sahi_tiled_v9_15k_iters_new_rotations
#SBATCH --exclude=gpu175,gpu150


source ~/.bashrc
source activate mask_rcnn
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# LazyConfig Training Script - pretrained new baseline
TILE_SIZE=512
python tools/lazyconfig_train_net.py --num-gpus 4 \
--resume \
--config-file /h/jquinto/Mask-RCNN/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--exp_id ${TILE_SIZE} \
--dataset_path /h/jquinto/MaskDINO/datasets/lifeplan_${TILE_SIZE}/ \
train.output_dir=output_${TILE_SIZE}_sahi_tiled_v9 \
dataloader.train.dataset.names=lifeplan_${TILE_SIZE}_train \
dataloader.test.dataset.names=lifeplan_${TILE_SIZE}_valid \

# # LazyConfig Training Script - from scratch
# python tools/lazyconfig_train_net.py --num-gpus 4 \
# --resume \
# --config-file /h/jquinto/Mask-RCNN/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --dataset_path /h/jquinto/MaskDINO/datasets/lifeplan_${TILE_SIZE}/ \
# train.output_dir=output_${TILE_SIZE}_sahi_tiled_v9_scratch \
# train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
# dataloader.train.dataset.names=lifeplan_${TILE_SIZE}_train \
# dataloader.test.dataset.names=lifeplan_${TILE_SIZE}_valid \
