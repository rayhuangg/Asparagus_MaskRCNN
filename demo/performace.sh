#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/semi_data/*.jpg --output /home/ubuntu/detectron2/demo/semi_data_output/ --json_output /home/ubuntu/detectron2/demo/semi_data_output/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/model_final.pth

# CUDA_VISIBLE_DEVICES=2,3 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/val/JPEGImages/*.jpg --output /home/ubuntu/detectron2/demo/performance_20221023_student/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/20220528_model_0aug_finetune_6/model_0003199.pth

# CUDA_VISIBLE_DEVICES=2,3 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/val/JPEGImages/*.jpg --output /home/ubuntu/detectron2/demo/performance_20221023_plain/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/pretrain_model_justin/model_final.pth