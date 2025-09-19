import logging
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
import cv2
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = get_cfg()
cfg.set_new_allowed(True) 
cfg = LazyConfig.load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py") 
# cfg.train.init_checkpoint = "/h/jquinto/Mask-RCNN/model_final_14d201.pkl"

# set model device
# cfg.MODEL.DEVICE = "self.device.type"
cfg.train.device = "cuda"

# # set input image size
# # NEW TRAINING PIPELINE
# cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.INPUT.MAX_SIZE_TEST = 1024
# cfg.freeze()

# init predictor
model = instantiate(cfg.model)
DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_lifeplan_b_512_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters/model_final.pth")
category_mapping={"1": "b"}

# detectron2 category mapping
category_names = list(category_mapping.values())
image = np.array(
    cv2.imread(
        '/home/mrajaraman/dataset/originals/img_1458477504.jpg', 
        cv2.IMREAD_COLOR
))

if isinstance(image, np.ndarray) and cfg.dataloader.train.mapper.image_format == "BGR":
        # convert RGB image to BGR format
        image = image[:, :, ::-1]
height, width = image.shape[:2]
mapper = instantiate(cfg.dataloader.test.mapper)
aug = mapper.augmentations
image = aug(T.AugInput(image)).apply_image(image)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
image = image.to(device)

model.to(device)
model.eval()

with torch.no_grad():
        inputs = {"image": image, "height": height, "width": width}
prediction_result = model([inputs])[0]
# print(prediction_result)

img = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
bbox = []
print("Number of detected instances is ", len(prediction_result['instances']._fields['pred_boxes'].tensor))
for i in prediction_result['instances']._fields['pred_boxes'].tensor:
        print("i is ", i)
        bbox.append(i.cpu().detach().numpy())

# Display the image
plt.imshow(Image.open('/home/mrajaraman/dataset/originals/img_1458477504.jpg'))

# Add the patch to the Axes
for i in range(len(bbox)):
        # print("length is ", len(bbox))
        plt.gca().add_patch(Rectangle((bbox[i][0], bbox[i][1]), bbox[i][2]-bbox[i][0], bbox[i][3]-bbox[i][1], linewidth=1, edgecolor='r', facecolor='none'))
        print(f"Instance {i+1} added to image")

plt.title("Inference Result of MaskRCNN on image")
plt.savefig("trained_inference_maskrcnn.png")

# image = np.ascontiguousarray(image).copy()
# img = torch.from_numpy(image)
# img = img.permute(2, 0, 1)  # HWC -> CHW

# if torch.cuda.is_available():
#         img = img.cuda()
# inputs = [{"image": img}]

# # run the model
# model.to(device)
# model.eval()
# with torch.no_grad():
#         predictions_ls = model(inputs)
# prediction_result = predictions_ls[0]
# print(prediction_result)
# original_predictions = prediction_result

# PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
print(aug)
print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
print(sample_preds)
print(sample_preds.shape)
print(sample_preds.dtype)

"""
>>> print(prediction_result['instances'][0]._fields['pred_masks'])
tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         ...,
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0')
>>> print(prediction_result['instances'][0]._fields['pred_masks'].size())
torch.Size([1, 256, 256])

- Note: we can see here that the output mask predictions are a float32 array that 
        is the same size as the sample image that we are predicting on.
>>> np.unique(sample_preds)
array([0., 1.], dtype=float32)
- Appears to be in RLE format -> indicating we have an RLE mask that by definition is pixel-based,
so we can't have sub-pixel mask coordinates:

The process is:

Upsample by 5x and round to nearest integer using +.5 trick
Get dense boundary points at this higher resolution
Downsample back by dividing by scale
Apply floor/ceil and boundary checks
Convert to final integer coordinates
So decimal coordinates are first scaled up for better precision during boundary calculation, but ultimately get converted to integers through this upscale-then-downscale process with rounding.

This explains why super-resolution could help - it effectively increases the resolution at which this rounding occurs, allowing for more precise boundary definitions.


"""