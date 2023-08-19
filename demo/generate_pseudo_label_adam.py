# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import csv
import tqdm
import pickle
import base64
import json
import numpy as np
from PIL import Image
from skimage import measure, morphology
from pycocotools import mask

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
# from plantcv import plantcv as pcv
# pcv.params.bebug = 'print'

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg





def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--json_output",
        help="Directory for json result output."
        )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def json_output(output, predictions, filename, path):
    '''
    Save predictions as json file
    '''
    out_filename = os.path.join(output, filename) + '.json'
    with open(path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    # labels = {1: 'clump', 2: 'stalk' , 3: 'spear', 4: 'bar'}
    labels = {1: 'stalk' , 2: 'spear'}
    image_height, image_width = predictions['instances'].image_size
    # pred_boxes = np.asarray(predictions["instances"].pred_boxes)
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()

    content = {
    "version": "4.5.5",
    "flags": {},
    "shapes": [
    ],
    "imagePath": filename,
    "imageData": img_data,
    "imageHeight": image_height,
    "imageWidth": image_width
    }

    for i in range(len(pred_classes)):
#        if pred_classes[i] == 1:
#            bbox = pred_boxes[i].cpu().numpy().tolist()
#            shape ={
#            "label": "clump",
#            "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
#            "group_id": i,
#            "shape_type": "rectangle",
#            "flags": {}
#            }
#            content["shapes"].append(shape)
#        else:
        segmentation = pred_masks[i]
        segmentation = measure.find_contours(segmentation.T, 0.5)
        for seg in segmentation:
            new_seg = []
            if len(seg) > 30:
                for k in range(0, len(seg), int(len(seg)/30)):
                    new_seg.append(seg[k].tolist())
            else:
                new_seg = [s.tolist() for s in seg]
            shape = {
            "label": labels[pred_classes[i]],
            "points": new_seg,
            "group_id": i,
            "shape_type": "polygon",
            "flags": {}
            }
            content["shapes"].append(shape)
    with open(out_filename, 'w+') as jsonfile:
        json.dump(content, jsonfile, indent=2)

def mask2skeleton(pred_mask):
    """
    Convert pred_masek into skeleton.

    Parameters:
        pred_mask(list): 2d list of boolean, Fasle for bg, True for fg.

    Returns:
        skeleton(list): 2d list of 0 and 1, o for bg, 1 for skeleton
    """
    pred_mask = np.asarray(pred_mask) + 0
    skeleton = morphology.skeletonize(pred_mask)
    return skeleton

def csv_out(predictions, filename, path):
    labels = {1: 'clump', 2: 'stalk' , 3: 'spear'}
    image_height, image_width = predictions['instances'].image_size
    pred_boxes = np.asarray(predictions["instances"].pred_boxes)
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()
    table = []
    for i in range(len(pred_classes)):
        if pred_classes[i] == 3:
            boxs = pred_boxes[i].cpu().numpy().tolist()
            skeleton = mask2skeleton(pred_masks[i]).tolist()
            length_skeleton = np.count_nonzero(skeleton)
            length_box = boxs[3] - boxs[1]
            table.append([i, boxs[0], boxs[1], boxs[2], boxs[3], mask.encode(np.asfortranarray(pred_masks[i])), length_skeleton, length_box])
    table = sorted(table, key = lambda x: x[1])
    for i in range(len(table)):
        table[i][0] = i
    table = [['id', 'box(xmin)', 'box(ymin)', 'box(xmax)', 'box(ymax)', 'mask', 'length(skeleton)', 'length(box)']] + table
    with open(os.path.join(path, filename) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
                filename = path.split('/')[-1][:-4]

                # pred_classes = predictions['instances'].pred_classes.cpu().numpy()
                # pred_scores = predictions['instances'].scores.cpu().numpy()
                # pred_boxes = np.asarray(predictions["instances"].pred_boxes)
                # pred_masks = predictions["instances"].pred_masks.cpu().numpy()
                # pred_all = []
                # for i in range(len(pred_classes)):
                #     try:
                #         pred_all.append([pred_classes[i], pred_scores[i], pred_boxes[i], pred_masks[i]])
                #     except:
                #         pass
                # pred_all = sorted(pred_all, key=lambda x: x[0])

                # pred_classes, pred_scores, pred_boxes, pred_masks = [], [], [], []
                # for i in range(len(pred_all)):
                #     pred_classes.append(pred_all[i][0])
                #     pred_scores.append(pred_all[i][1])
                #     pred_boxes.append(pred_all[i][2])
                #     pred_masks.append(pred_all[i][3])

                # for i in range(len(pred_classes)):
                #     if pred_classes[i] == 3:
                #         segmentation = pred_masks[i].astype(np.uint8)
                #         props = measure.regionprops(segmentation)
                #         for prop in props:
                #             print(i)
                #             print('Length:', prop.major_axis_length)
                #             print('Width: ', prop.minor_axis_length)
                #             print('---')
                #         skeleton = pcv.morphology.skeletonize(mask=segmentation)
                #         segmented_img, obj = pcv.morphology.segment_skeleton(skel_img=skeleton)
                #         labeled_img = pcv.morphology.segment_path_length(segmented_img=segmented_img, objects=obj, label="default")
                #         path_lengths = pcv.outputs.observations['default']['segment_path_length']['value']
                #         height = np.count_nonzero(skeleton)
                #         widths = []
                #         for j, row in enumerate(skeleton):
                #             for k, p in enumerate(row):
                #                 if p:
                #                     widths.append(distance_transformation[j, k])
                #         width = round((sum(widths)/len(widths)), 2)
                #         print('skeleton:', type(skeleton))
                #         cv2.imwrite(os.path.join(args.output, 'spear_'+filename+'_'+str(i)+'.jpg'), segmented_img)
                #         print(obj)
                #         print('height: ', height)
                #         print('path length:', path_lengths)
                #         print('score: ', pred_scores[i])
                #         print('width:', width)

                if args.json_output:
                    json_output(args.json_output, predictions, filename, path)
