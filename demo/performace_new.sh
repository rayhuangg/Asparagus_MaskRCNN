#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/val/JPEGImages/*.jpg --output /home/ubuntu/detectron2/demo/performace_new/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/model_final.pth