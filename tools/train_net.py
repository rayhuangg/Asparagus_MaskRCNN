#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from datetime import date, datetime
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from detectron2.utils.events import TensorboardXWriter
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import detectron2.data.transforms as T

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[
            T.ResizeShortestEdge(short_edge_length=[900, 1200], max_size=1600, sample_style='range'),
            T.RandomFlip(),
            # T.RandomRotation(5, expand = False),
            # T.RandomCrop('relative_range', (0.5, 0.5)),
            T.RandomContrast(0.8, 1.2),
            T.RandomBrightness(0.8, 1.2),
        #     T.RandomExtent((0.2, 0.8), (0.2, 0.2))
        ]))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.eval_only:
        train_outdir = cfg.OUTPUT_DIR
        cfg.OUTPUT_DIR = f"{train_outdir}_evaluation_time-{date}"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_my_dataset():
    #========= Register COCO dataset =========
    metadata = {"thing_classes": ["stalk", "spear"],
                "thing_colors": [(41,245,0), (200,6,6)]}
    metadata_background = {"thing_classes": ["_background_","stalk", "spear"], # unknown reason: MaskRCNN label will from 1 to 2
                           "thing_colors": [(255,255,255), (41,245,0), (200,6,6)]}

    # small test
    # register_coco_instances('asparagus_train_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_train2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")

    # full data
    # register_coco_instances('asparagus_train_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_train2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_full', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230803_test_small_dataset_with_background_class/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val', metadata_background, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230627_Adam_ver/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")

    # Adam raw valdation
    # register_coco_instances('asparagus_val', {'_background_': 0, 'stalk': 1, 'spear': 2}, "/home/rayhuang/Asparagus_Dataset/val/annotations.json", "/home/rayhuang/Asparagus_Dataset/val")
    register_coco_instances('asparagus_val', {}, "/home/rayhuang/Asparagus_Dataset/val/annotations.json", "/home/rayhuang/Asparagus_Dataset/val")


def main(args):
    register_my_dataset()
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
