# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import List, Optional
import contextlib

import numpy as np
import torch

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

logger = logging.getLogger(__name__)


class Detectron2DetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "detectron2"])

    def load_model(self):
        # from detectron2.config import get_cfg
        # from detectron2.data import MetadataCatalog
        # from detectron2.engine import DefaultPredictor
        # from detectron2.model_zoo import model_zoo

        cfg = get_cfg()
        cfg.set_new_allowed(True) 
        cfg = LazyConfig.load(self.config_path) 
        self.cfg = cfg
        cfg.train.init_checkpoint = self.model_path 

        # set model device
        cfg.train.device = self.device.type

        # set input image size
        # if self.image_size is not None:
        #     cfg.INPUT.MIN_SIZE_TEST = self.image_size
        #     cfg.INPUT.MAX_SIZE_TEST = self.image_size
        
        # init predictor
        # model = DefaultPredictor(cfg)
        model = instantiate(cfg.model)
        DetectionCheckpointer(model).load(self.model_path)

        self.model = model

        # detectron2 category mapping
        if self.category_mapping is None:
            try:  # try to parse category names from metadata
                metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names)
                category_names = metadata.thing_classes
                self.category_names = category_names
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
            except Exception as e:
                logger.warning(e)
                # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
                num_categories = cfg.model.roi_heads.num_classes
                self.category_names = [str(category_id) for category_id in range(num_categories)]
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
        else:
            self.category_names = list(self.category_mapping.values())

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        if isinstance(image, np.ndarray) and self.cfg.dataloader.train.mapper.image_format == "BGR":
                # convert RGB image to BGR format
                image = image[:, :, ::-1]
        height, width = image.shape[:2]

        @contextlib.contextmanager
        def disable_detectron2_logging(level=logging.ERROR):
            """Context manager to temporarily change detectron2 logging level."""
            logger = logging.getLogger("detectron2")
            old_level = logger.level
            logger.setLevel(level)
            try:
                yield
            finally:
                logger.setLevel(old_level)
        with disable_detectron2_logging():
            mapper = instantiate(self.cfg.dataloader.test.mapper)

        aug = mapper.augmentations
        image = aug(T.AugInput(image)).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image = image.to(self.device)

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            inputs = {"image": image, "height": height, "width": width}
        prediction_result = self.model([inputs])[0]
        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor
        scores = original_predictions["instances"].scores
        category_ids = original_predictions["instances"].pred_classes

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks
        except AttributeError:
            masks = None

        # filter predictions with low confidence
        high_confidence_mask = scores >= self.confidence_threshold
        boxes = boxes[high_confidence_mask]
        scores = scores[high_confidence_mask]
        category_ids = category_ids[high_confidence_mask]
        if masks is not None:
            masks = masks[high_confidence_mask]
        if masks is not None:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist() if mask is None else None,
                    segmentation=get_coco_segmentation_from_bool_mask(mask.detach().cpu().numpy())
                    if mask is not None
                    else None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id, mask in zip(boxes, scores, category_ids, masks)
                if mask is None or ((segmentation := get_coco_segmentation_from_bool_mask(mask.detach().cpu().numpy())) and len(segmentation) > 0) and get_bbox_from_bool_mask(mask.detach().cpu().numpy()) is not None
            ]
        else:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist(),
                    segmentation=None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id in zip(boxes, scores, category_ids)
            ]

        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image
