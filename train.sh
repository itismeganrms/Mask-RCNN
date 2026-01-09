#!/bin/bash

#SBATCH --job-name=maskrcnn
#SBATCH --time=18:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --output=/home/mrajaraman/slurm/maskrcnn/merged-train/output-%A.out
#SBATCH --gres=gpu:1

echo "## Starting GPU test on $HOSTNAME"
module purge

echo "## Loading module"
module load ALICE/default
module load Miniconda3
conda init

source activate /home/mrajaraman/conda/mask_rcnn

# Debugging outputs
pwd
which conda
python --version
# pip freeze

# LazyConfig Training Script - pretrained new baseline  
# ---- TRAINING COMMANDS FOR ORIGINAL DATASET BELOW ----
TILE_SIZE=512
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# python /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/tools/lazyconfig_train_net.py --num-gpus 1 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --train_iter 2500 \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# train.output_dir=output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
# dataloader.train.dataset.names=dragonfly_512_train \
# dataloader.test.dataset.names=dragonfly_512_valid 
# # dataloader.test.dataset.names=dragonfly_512_test \

# python /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/tools/lazyconfig_train_net.py --num-gpus 1 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --train_iter 7500 \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# train.output_dir=output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
# dataloader.train.dataset.names=dragonfly_512_train \
# dataloader.test.dataset.names=dragonfly_512_valid \

# python /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/tools/lazyconfig_train_net.py --num-gpus 1 \
# --config-file /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --train_iter 15000 \
# --dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
# train.output_dir=output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
# train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
# dataloader.train.dataset.names=dragonfly_512_train \
# dataloader.test.dataset.names=dragonfly_512_valid 

# # ---- TRAINING COMMANDS FOR MERGED DATASET BELOW ----
python /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/tools/lazyconfig_train_net_merged.py --num-gpus 1 \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--exp_id ${TILE_SIZE} \
--train_iter 14550 \
--dataset_path /home/mrajaraman/dataset/dataset-v2-coco/ \
train.output_dir=output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
dataloader.train.dataset.names=dragonfly_merged_train \
dataloader.test.dataset.names=dragonfly_merged_valid 