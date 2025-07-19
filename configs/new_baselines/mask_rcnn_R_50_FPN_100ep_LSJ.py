import detectron2.data.transforms as T
from detectron2.config.lazy import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler
from PIL import Image

from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.optim import AdamW as optimizer
from ..common.train import train

# train from scratch
# train.init_checkpoint = "/h/jquinto/Mask-RCNN/model_final_f96b26.pkl" # R101
# train.init_checkpoint = "/h/jquinto/Mask-RCNN/model_final_f1362d.pkl" # RegNetX
train.init_checkpoint = "/h/jquinto/Mask-RCNN/model_final_14d201.pkl" # R50 model
train.amp.enabled = True
train.ddp.fp16_compression = True
train.checkpointer=dict(period=4885, max_to_keep=100)  # options for PeriodicCheckpointer
train.eval_period=100000
model.backbone.bottom_up.freeze_at = 0

# SyncBN
# fmt: off
model.backbone.bottom_up.stem.norm = \
    model.backbone.bottom_up.stages.norm = \
    model.backbone.norm = "SyncBN"

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.box_head.conv_norm = \
    model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c,
                                                                       stats_mode="N")
# fmt: on

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# resize_and_crop_image in:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/utils/input_utils.py#L127  # noqa: E501, B950
image_size = 1024
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.RandomRotation)(angle=[0, 90, 180, 270], sample_style = 'choice'),
    L(T.RandomBrightness)(intensity_min=0.85, intensity_max=1.15),
    L(T.RandomContrast)(intensity_min=0.9, intensity_max=1.1),
    L(T.RandomSaturation)(intensity_min=0.85, intensity_max=1.15),
    L(T.ResizeShortestEdge)(short_edge_length=(image_size, image_size), max_size = image_size, sample_style = 'choice', interp=Image.BILINEAR),
]

# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

# larger batch-size.
dataloader.train.total_batch_size = 8

train.max_iter = 15000

def cosine_lr_scheduler(
    start_value,
    end_value,
    num_updates,
    warmup_steps,
    warmup_method="linear",
    warmup_factor=0.001,
):
    
    # define cosine scheduler
    scheduler = L(CosineParamScheduler)(
        start_value=start_value,
        end_value=end_value,
    )

    # wrap with warmup scheduler
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / num_updates,
        warmup_method=warmup_method,
        warmup_factor=warmup_factor,
    )

BASE_LR = 5e-5
# define cosine scheduler
lr_multiplier = cosine_lr_scheduler(
    start_value=1,
    end_value=0.0001,
    num_updates=int(train.max_iter),
    warmup_steps=int(0.3*train.max_iter),
    warmup_method="linear",
    warmup_factor=0.001
)
optimizer.lr = BASE_LR
optimizer.weight_decay = 0.05
