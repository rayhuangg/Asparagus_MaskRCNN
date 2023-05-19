#!/bin/bash

python3 demo.py \
    --config-file /home/ubuntu/detectron2/output/pretrain_model_justin/config.yaml \
    --input /home/ubuntu/detectron2/datasets/coco/annotations2/val_old/JPEGImages/*.jpg \
    --output /home/ubuntu/detectron2/output_predict_justin/ \
    --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/pretrain_model_justin/model_final.pth