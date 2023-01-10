# CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 2 --resume MODEL.WEIGHTS "/home/ubuntu/.torch/iopath_cache/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl" SOLVER.IMS_PER_BATCH 6

# CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 3 --resume MODEL.WEIGHTS "/home/ubuntu/detectron2/output/20220503_teacher_model/model_final.pth" SOLVER.IMS_PER_BATCH 6

# CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 3 --resume MODEL.WEIGHTS "/home/ubuntu/detectron2/output/20220503_teacher_model/model_final.pth" SOLVER.IMS_PER_BATCH 6

# CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 3 --resume MODEL.WEIGHTS "/home/ubuntu/detectron2/output/20220503_teacher_model/model_final.pth" SOLVER.IMS_PER_BATCH 6

# 20230107 test
CUDA_VISIBLE_DEVICES=1 python3 tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --num-gpus 1 --resume MODEL.WEIGHTS "/home/ubuntu/.torch/iopath_cache/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl" SOLVER.IMS_PER_BATCH 6
