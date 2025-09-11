#!/bin/bash

#SBATCH --job-name=maskrcnn
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/home/mrajaraman/slurm/maskrcnn/train/output-%A.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100.4g.40gb|A100.3g.40gb"


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
TILE_SIZE=512

# python /home/mrajaraman/master-thesis-dragonfly/mask-rcnn/Mask-RCNN-MassID45/tools/lazyconfig_train_net.py --num-gpus 2 \
# --resume \
# --config-file /home/mrajaraman/master-thesis-dragonfly/mask-rcnn/Mask-RCNN-MassID45/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
# --exp_id ${TILE_SIZE} \
# --dataset_path /home/mrajaraman/dataset/first_batch_pngs/data/ \
# train.output_dir=output_${TILE_SIZE}_megan\
# dataloader.train.dataset.names=/home/mrajaraman/do-not-modify/image-to-coco-json-converter/output/train.json \
# dataloader.test.dataset.names=/home/mrajaraman/do-not-modify/image-to-coco-json-converter/output/val.json \

# # LazyConfig Training Script - from scratch

python /home/mrajaraman/master-thesis-dragonfly/mask-rcnn/Mask-RCNN-MassID45/tools/lazyconfig_train_net.py --num-gpus 1 \
--resume \
--config-file /home/mrajaraman/master-thesis-dragonfly/mask-rcnn/Mask-RCNN-MassID45/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--exp_id ${TILE_SIZE} \
--dataset_path /home/mrajaraman/dataset/coco/ \
train.output_dir=output_${TILE_SIZE}_sahi_tiled_v9_scratch \
train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
dataloader.train.dataset.names=dragonfly_512_train \
dataloader.test.dataset.names=dragonfly_512_valid \
# dataloader.val.dataset.names=dragonfly_512_valid \
