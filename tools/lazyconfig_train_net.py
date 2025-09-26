import logging
import wandb
import detectron2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import EventWriter

import os
import random 
logger = logging.getLogger("detectron2")

run = wandb.init(
    entity="universiteitleiden",
    project="master-thesis-dragonfly",
    tags=["Mask-R-CNN", "annotated-images", "4-part-annotated" ],
    config={
        "architecture": "Mask-R-CNN",
    },
    sync_tensorboard=True
)

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret

def do_train(args, cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    cfg.train.max_iter = args.train_iter  
    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    num_ckpts = 10
    period_steps = max(1, cfg.train.max_iter // num_ckpts)
    cfg.train.checkpointer=dict(period=period_steps, max_to_keep=10) 
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

from detectron2.data.datasets import register_coco_instances

def register_custom_coco_dataset(args) -> None:
   dataset_path = args.dataset_path
   exp_id = args.exp_id
   annotations_path = os.path.join(dataset_path, "annotations/")
   register_coco_instances(
       f"dragonfly_{exp_id}_train",
       {},
       os.path.join(annotations_path, "instances_train.json"),
       os.path.join(dataset_path, "train"),
   )
   if args.eval_only:
    register_coco_instances(
        f"dragonfly_{exp_id}_test",
        {},
       os.path.join(annotations_path, "instances_test.json"),
       os.path.join(dataset_path, "test"), ## NOTE: we generally do not want to test on the tiled test set
    )
   else: 
    register_coco_instances(
        f"dragonfly_{exp_id}_valid",
        {},
        os.path.join(annotations_path, "instances_val.json"),
        os.path.join(dataset_path, "valid"),
    )

def main(args):
    register_custom_coco_dataset(args) # REGISTER CUSTOM COCO DATA -> disable during inference
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


def invoke_main() -> None:
    args = default_argument_parser().parse_args()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--dataset_path', type=str, default="/home/mrajaraman/dataset/coco-roboflow/", help="Path to the dataset directory containing annotations and images")
    parser.add_argument('--exp_id', type=int, default=256, help="Identifier string -- tile size for training models")
    parser.add_argument('--train_iter', type=int, default=2000, help="Number of iterations to train for")
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

# if __name__ == "__main__":
#     invoke_main() 
