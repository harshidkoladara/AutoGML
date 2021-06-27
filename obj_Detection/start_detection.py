import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
try:
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized
except:
    from obj_Detection.utils.datasets import LoadStreams, LoadImages
    from obj_Detection.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from obj_Detection.utils.plots import plot_one_box
    from obj_Detection.utils.torch_utils import select_device, load_classifier, time_synchronized

from .detect import detect


class Opt:
    def __init__(self, weights, source, img_size, conf, device, project, name):
        self.weights = [weights]
        self.source = source
        self.img_size = img_size
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.nosave = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = project
        self.name = name
        self.exist_ok = False


def start_detection(weights, source, img_size, conf, device, project, name):
    # opt = parser.parse_args()
    opt = Opt(weights, source, img_size, conf, device, project, name)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)
