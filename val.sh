#!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file /home/ubuntu/detectron2/output/config.yaml --num-gpu 2 --dist-url tcp://0.0.0.0:50515 --eval-only MODEL.WEIGHTS ./output/model_0194999.pth SOLVER.IMS_PER_BATCH 1

# CUDA_VISIBLE_DEVICES=2 python3 tools/train_net.py --config-file /home/ubuntu/detectron2/output/20220527_model/config.yaml --num-gpu 1 --dist-url tcp://0.0.0.0:50515 --eval-only MODEL.WEIGHTS ./output/model_0209999.pth SOLVER.IMS_PER_BATCH 1

# CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file /home/ubuntu/detectron2/output/config.yaml --num-gpu 2 --dist-url tcp://0.0.0.0:50515 --eval-only MODEL.WEIGHTS ./output/model_0244999.pth SOLVER.IMS_PER_BATCH 1

# CUDA_VISIBLE_DEVICES=2,3 python3 tools/train_net.py --config-file /home/ubuntu/detectron2/output/config.yaml --num-gpu 2 --dist-url tcp://0.0.0.0:50515 --eval-only MODEL.WEIGHTS ./output/model_0249999.pth SOLVER.IMS_PER_BATCH 1

# 20230112 Testing
CUDA_VISIBLE_DEVICES=1 python3 tools/train_net.py --config-file output/20230111/config.yaml --num-gpu 1 --dist-url tcp://0.0.0.0:50515 --eval-only MODEL.WEIGHTS output/20230111/model_final.pth