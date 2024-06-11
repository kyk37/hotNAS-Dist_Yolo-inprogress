#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Miscellaneous utility functions."""

import os
import numpy as np
import time
import cv2, colorsys
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from common.backbones.efficientnet import swish
from common.backbones.mobilenet_v3 import hard_sigmoid, hard_swish

#from yolo4.models.layers import mish
#import tensorflow as tf

import torch
from torch import nn

def optimize_pytorch_gpu(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #enable all GPUs to be used
    model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids = [0,1])
    model.to(device)
    return device

# def optimize_tf_gpu(tf, K):
#     if tf.__version__.startswith('2'):
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 # Currently, memory growth needs to be the same across GPUs
#                 for gpu in gpus:
#                     # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
#                     tf.config.experimental.set_memory_growth(gpu, True)
#             except RuntimeError as e:
#                 # Memory growth must be set before GPUs have been initialized
#                 print(e)
#     else:
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True  # dynamic alloc GPU resource
#         config.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU memory threshold 0.3
#         session = tf.Session(config=config)

#         # set session
#         K.set_session(session)


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,# Tensorflow
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }

    return custom_objects_dict

def mish(x):
    return x * torch.tanh(nn.Softplus(x))



def get_multiscale_list():
    input_shape_list = [(320, 320), (352, 352), (384, 384), (416, 416), (448, 448), (480, 480), (512, 512), (544, 544),
                        (576, 576), (608, 608)]

    return input_shape_list


def resize_anchors(base_anchors, target_shape, base_shape=(416, 416)):
    '''
    original anchor size is clustered from COCO dataset
    under input shape (416,416). We need to resize it to
    our train input shape for better performance
    '''
    return np.around(base_anchors * target_shape[::-1] / base_shape[::-1])


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def get_colors_fixed_map():
    colors = {
        "Pedestrian": (255, 0, 0),
        "Car": (255, 255, 0),
        "Van": (255, 255, 255),
        "Truck": (0, 255, 255),
        "Person_sitting": (0, 0, 255),
        "Cyclist": (255, 0, 255),
        "Tram": (0, 0, 0)
    }
    return colors


def get_dataset(annotation_file, dataset_working_directory, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for i in range(len(lines)):
            tmp = lines[i].split(" ")
            tmp[0] = os.path.join(dataset_working_directory, tmp[0])
            lines[i] = str.join(" ", tmp)

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)

    return lines


def _draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    label_color = (abs(color[0] - 255), abs(color[1] - 255), abs(color[2] - 255))

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=label_color,
                lineType=cv2.LINE_AA)

    return image


def _draw_boxes3(image, pred_boxes, pred_classes, colors, gt_boxes=None, drawn_labels=None):
    if pred_classes is None or len(pred_classes) == 0:
        return image

    for i in range(0, len(pred_boxes)):
        if gt_boxes is not None:
            color = (128, 128, 128)
            xmin, ymin, xmax, ymax = gt_boxes[i]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)

        color = colors[i]
        xmin, ymin, xmax, ymax = pred_boxes[i]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)

        if drawn_labels is not None:
            text = drawn_labels[i]
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1.
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
            label_color = (abs(color[0] - 255), abs(color[1] - 255), abs(color[2] - 255))

            padding = 5
            rect_height = text_height + padding * 2
            rect_width = text_width + padding * 2

            (x, y) = (xmin, ymin)
            cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
            cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                        fontScale=font_scale,
                        color=label_color,
                        lineType=cv2.LINE_AA)
    return image