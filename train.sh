#!/bin/bash

#SBATCH --job-name=maskrcnn
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/home/mrajaraman/slurm/maskrcnn/new-train/output-%A.out
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
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

python /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/tools/lazyconfig_train_net.py --num-gpus 1 \
--config-file /home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
--exp_id ${TILE_SIZE} \
--train_iter 2000 \
--dataset_path /home/mrajaraman/dataset/coco-roboflow/ \
train.output_dir=output_${TILE_SIZE}_dragonfly_${TIMESTAMP} \
train.init_checkpoint="detectron2://ImageNetPretrained/torchvision/R-50.pkl" \
dataloader.train.dataset.names=dragonfly_512_train \
dataloader.test.dataset.names=dragonfly_512_valid \
# dataloader.test.dataset.names=dragonfly_512_test \