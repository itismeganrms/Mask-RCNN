import logging
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
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
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_0004884.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_0009769.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_0014654.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_0019539.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_0024424.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_sahi_tiled_v9_scratch/model_final.pth")

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0000199.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0000399.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0000599.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0000799.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0000999.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0001199.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0001399.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0001599.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0001799.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_0001999.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-50-03/model_final.pth")

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-28-53/model_0001999.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-28-53/model_final.pth")

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-22_01-04-50/model_final.pth")

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-21_03-09-31/model_final.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_00-35-53_still_glitter_192/model_final.pth")
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_00-50-59_amber_grass_194/model_final.pth")


# category_mapping={"0": "head", "1": "torso", "2": "tail", "3": "wings"}
# category_mapping={"0": "wings", "1": "head", "2": "thorax", "3": "abdomen"} ## OLD INFERENCE
# category_mapping={0: "background", 1: "head", 2": "abdomen", "3": "thorax", "4":"wings"}}

## OLD INFERENCE SNIPPET
# dataset_name="dataset_v1_coco"
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_00-35-53_still_glitter_192/model_final.pth")
# model_name="model_old_2500"

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-09-28_13-30-18_lyric_dawn_144/model_0004999.pth")
# model_name="model_old_5000"

# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_00-50-59_amber_grass_194/model_final.pth")
# model_name="model_old_15000"
# MetadataCatalog.get(dataset_name).thing_classes = ["dragonfly", "head", "abdomen", "thorax", "wings"]

# ## NEW INFERENCE SNIPPET
# dataset_name="dataset_v2_coco"
# DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/output_512_dragonfly_2025-12-15_15-48-49_fine_fire_199/model_final.pth")
# MetadataCatalog.get(dataset_name).thing_classes = ["objects", "head", "abdomen", "thorax", "wings"]
# model_name="model_final_new_trained"

# detectron2 category mapping
# category_names = list(category_mapping.values())
image = np.array(cv2.imread('/home/mrajaraman/dataset/originals/img_1458477504.jpg', cv2.IMREAD_COLOR))
np_image = image.copy()

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


# model_name="model_old_2500"

# print("Inference done on ", model_name)

with torch.no_grad():
        inputs = {"image": image, "height": height, "width": width}
prediction_result = model([inputs])[0]
print(prediction_result)

# print(aug)
print()

# PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
print(aug)
print()

outputs = prediction_result["instances"]
# instances = outputs["instances"].to("cpu")
# instances.pred_classes = instances.pred_classes.clone()

# print(type(outputs))
# print(type(image))

# v = Visualizer(np_image[:, :, ::-1], metadata=None, scale=1.0)
# out = v.draw_instance_predictions(outputs.to("cpu"))
v = Visualizer(np_image[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1.0)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
out = v.draw_instance_predictions(outputs.to("cpu"))

plt.figure(figsize=(8, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.title("Inference of Trained Mask-RCNN on Image", fontsize=16)
plt.savefig(f"/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/final_runs_for_consideration/all_inference_images/inference_{model_name}.png",dpi=300, bbox_inches="tight")
plt.show()