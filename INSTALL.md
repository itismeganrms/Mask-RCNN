## Installation
Follow the instructions below to set-up the conda environment:

### Example conda environment setup
```bash
conda create --name mask_rcnn python=3.8 -y
conda activate mask_rcnn
# optional: module load cuda-11.3 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install opencv-python==4.9.0.80

# Navigate to Mask2Former directory
cd Mask-RCNN

# Install prebuilt detectron2 - see https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Move the following files to the location of your conda environments (.e.g, miniconda/envs/maskdino/lib/python3.8/site-packages/detectron2/)

# Implements patch for lazyconfig -- see PR: https://github.com/facebookresearch/detectron2/pull/3755/files#diff-882061b431ed3670f5b4a045ae0f4e1f140cc785db77fe77585b820bdec6f73d
cp detectron2_modifications/defaults.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/engine/defaults.py

cp detectron2_modifications/augmentation_impl.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/data/transforms/augmentation_impl.py

# Fix ensures images without annotations do not throw an error during training
cp detectron2_modifications/dataset_mapper.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/detectron2/data/dataset_mapper.py

# Install certain versions of packages to avoid errors later on
pip install numpy==1.23.1
pip install pillow==9.5.0
pip install sahi==0.11.18
pip install pycocotools

# Move the following files to miniconda/envs/maskdino/lib/python3.8/site-packages/sahi/:
cp sahi_modifications/detectron2.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/sahi/models/detectron2.py

cp sahi_modifications/annotation.py /miniconda/envs/mask_rcnn/lib/python3.8/site-packages/sahi/annotation.py

```
