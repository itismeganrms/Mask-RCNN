import logging
from typing import List, Optional

import numpy as np
import pandas as pd
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
from detectron2.data import MetadataCatalog
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

DetectionCheckpointer(model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-28_13-30-18/model_final.pth")

category_mapping={"0": "head", "1": "torso", "2": "tail", "3": "wings"}

test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/test_batch.csv")
category_names = list(category_mapping.values())


for image_path in test_images['Path']:
        image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))
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

        image_id = image_path[47:-4]
        print("Inference done on ", image_id)

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
        # print(type(outputs))
        # print(type(image))

        v = Visualizer(np_image[:, :, ::-1], metadata=None, scale=1.0)
        out = v.draw_instance_predictions(outputs.to("cpu"))
        plt.imshow(out.get_image())
        plt.axis("off")
        plt.title("Inference Result of MaskRCNN on image")
        plt.savefig(f"inferences_28/inference_{image_id}.png")
        plt.show()