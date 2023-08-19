#!/bin/bash

# ============== previous result =================

# CUDA_VISIBLE_DEVICES=0,1 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/semi_data/*.jpg --output /home/ubuntu/detectron2/demo/semi_data_output/ --json_output /home/ubuntu/detectron2/demo/semi_data_output/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/model_final.pth

# CUDA_VISIBLE_DEVICES=2,3 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/val/JPEGImages/*.jpg --output /home/ubuntu/detectron2/demo/performance_20221023_student/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/20220528_model_0aug_finetune_6/model_0003199.pth

# CUDA_VISIBLE_DEVICES=2,3 python3 demo_adam.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /home/ubuntu/detectron2/datasets/coco/annotations2/val/JPEGImages/*.jpg --output /home/ubuntu/detectron2/demo/performance_20221023_plain/ --opts MODEL.WEIGHTS /home/ubuntu/detectron2/output/pretrain_model_justin/model_final.pth

# ============== current result =================

CUDA_VISIBLE_DEVICES=1 python demo_adam.py \
    --json_output demo_pic/ \
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
    --input /home/rayhuang/Asparagus_Dataset/robot_regular_patrol/20211129/20211129_08_30_04_.jpg --output demo_pic/ \
    --opts MODEL.WEIGHTS /home/rayhuang/MaskRCNN/output/20220528_model_0aug_finetune_6/model_0007999.pth



# /home/rayhuang/Asparagus_Dataset/robot_regular_patrol/20211129/20211129_08_30_04_.jpg 1920*1080
# /home/rayhuang/Asparagus_Dataset/Adam_pseudo_label/Justin_remain/390.jpg 3280*2464
# /home/rayhuang/Asparagus_Dataset/Adam_pseudo_label/Justin_remain/667.jpg 4032*3024
# /home/rayhuang/Asparagus_Dataset/Justin_labeled_data/162.jpg  4592*3448
# /home/rayhuang/Asparagus_Dataset/Justin_labeled_data/309.jpg  5472*3648