# -*- coding=utf-8 -*-
#!/usr/bin/python3


## TODO: Convert this from Tensorflow to Pytorch
## Note verify all "expand_dims" conversion to "unsqueeze" or "squeeze"

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from yolo3.postprocess import yolo3_decode

import torch
from torch import nn
from torch import Tensor

def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute softmax focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
    """
    y_pred = nn.Softmax(y_pred)
    y_pred = torch.maximum(torch.minimum(y_pred, 1 - 1e-15), 1e-15)
    
    # Calculate Cross Entropy
    cross_entropy = -y_true * torch.log(y_pred)
    
    # Calculate Focal Loss
    softmax_focal_loss = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy
    
    return softmax_focal_loss



def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = nn.BCELoss(y_true, y_pred, from_logits=True)

    pred_prob = nn.Sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = torch.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    #sigmoid_focal_loss = torch.sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss

def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = b1.unsqueeze(-2) #b1 = torch.expand(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = b2.unsqueeze(0) #b2 = torch.expand(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half
    
    intersect_mins = torch.maximum(b1_mins, b2_mins)
    intersect_maxes = torch.minimum(b1_maxes, b2_maxes)
    intersect_wh = torch.clamp(intersect_maxes - intersect_mins, 0.) #torch.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b_true, b_pred):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = torch.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = torch.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = torch.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # get enclosed area
    enclose_mins = torch.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = torch.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = torch.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
    giou = giou.unsqeeze(-1) #K.expand_dims(giou, -1)

    return giou



def box_diou(b_true, b_pred, use_ciou=False):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    use_ciou: bool flag to indicate whether to use CIoU loss type

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = torch.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = torch.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = torch.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = torch.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = torch.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = torch.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = torch.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = torch.sum(torch.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        # calculate param v and alpha to extend to CIoU
        v = 4*torch.square(torch.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - torch.atan2(b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)

        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).
        v = v * torch.detach(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1]) # tf.stop_gradient()

        alpha = v / (1.0 - iou + v)
        diou = diou - alpha*v

    diou = diou.unsqueeze(-1) # K.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = torch.tensor(label_smoothing, dtype=torch.float32) #K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing



def yolo3_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, elim_grid_sense=False, use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=True):
    '''
    YOLOv3 loss function.

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    if num_layers == 3:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    input_shape = torch.tensor(yolo_outputs[0].shape[1:3] * 32 , dtype = y_true[0])  #K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [torch.tensor(yolo_outputs[l].shape[1:3], dtype=y_true[0]) for l in range(num_layers)] # [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    total_dist_loss = 0
    batch_size = yolo_outputs[0].shape[0] #K.shape(yolo_outputs[0])[0] # batch size, tensor
    batch_size_f = torch.tensor(batch_size, torch.dtype[yolo_outputs]) #K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:5+num_classes]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)
            true_objectness_probs = _smooth_labels(object_mask, label_smoothing)
        else:
            true_objectness_probs = object_mask

        grid, raw_pred, pred_xy, pred_wh = yolo3_decode(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, scale_x_y=scale_x_y[l], calc_loss=True)
        pred_box = torch.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = torch.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = torch.where(object_mask.bool(), raw_true_wh, torch.zeros_like(raw_true_wh)) #K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        raw_true_dist = y_true[l][..., 5+num_classes:5+num_classes+1]
        # Find ignore mask, iterate over each of batch.
        ignore_mask = [] #tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = torch.tensor(object_mask, dtype=torch.bool) #K.cast(object_mask, 'bool')
        
        # def loop_body(b, ignore_mask):
        #     true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
        #     iou = box_iou(pred_box[b], true_box)
        #     best_iou = K.max(iou, axis=-1)
        #     ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        #     return b+1, ignore_mask
        # _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
        
        for b in range(batch_size):
            true_box = torch.masked_select(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = torch.max(iou, axis=-1)[0] #may not need the [0] if error remove it
            ignore_mask.append((best_iou<ignore_thresh).float().unsqueeze(-1)) ####-- May need modifications
        
        ignore_mask = torch.stack(ignore_mask) #ignore_mask = ignore_mask.stack()
        ignore_mask = ignore_mask.unsqueeze(-1) #ignore_mask = K.expand_dims(ignore_mask, -1)


        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = sigmoid_focal_loss(true_objectness_probs, raw_pred[...,4:5])
        else:
            confidence_loss = object_mask * nn.functional.binary_cross_entropy(true_objectness_probs, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * nn.functional.binary_cross_entropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

        if use_focal_loss:
            # Focal loss for classification score
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_probs, raw_pred[...,5:5+num_classes])
            else:
                class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[...,5:5+num_classes])
        else:
            if use_softmax_loss:
                
                # use softmax style classification output
                # ------------ This one may need to be modified ----------------- #
                class_loss = object_mask * nn.functional.cross_entropy(raw_pred[..., 5:5 + num_classes], true_class_probs, reduction='none').unsqueeze(-1)  # object_mask * K.expand_dims(K.categorical_crossentropy(true_class_probs, raw_pred[...,5:5+num_classes], from_logits=True), axis=-1)
            else:
                # use sigmoid style classification output
                class_loss = object_mask * nn.functional.binary_cross_entropy(true_class_probs, raw_pred[...,5:5+num_classes], from_logits=True)


        if use_giou_loss:
            # Calculate GIoU loss as location loss
            raw_true_box = y_true[l][...,0:4]
            giou = box_giou(raw_true_box, pred_box)
            giou_loss = object_mask * box_loss_scale * (1 - giou)
            giou_loss = torch.sum(giou_loss) / batch_size_f #K.sum(giou_loss) / batch_size_f
            location_loss = giou_loss
        elif use_diou_loss:
            # Calculate DIoU loss as location loss
            raw_true_box = y_true[l][...,0:4]
            diou = box_diou(raw_true_box, pred_box)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = torch.sum(diou_loss) / batch_size_f #K.sum(diou_loss) / batch_size_f
            location_loss = diou_loss
        else:
            # Standard YOLOv3 location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * nn.functional.binary_cross_entropy(raw_true_xy, raw_pred[...,0:2], from_logits=True) #K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * torch.square(raw_true_wh-raw_pred[...,2:4]) #K.square
            xy_loss = torch.sum(xy_loss) / batch_size_f
            wh_loss = torch.sum(wh_loss) / batch_size_f
            location_loss = xy_loss + wh_loss

        dist_loss = object_mask * 0.5 * K.square(raw_true_dist - raw_pred[..., 5+num_classes:5+num_classes+1])

        dist_loss = torch.sum(dist_loss) / batch_size_f * 0.01 #K.sum(dist_loss) / batch_size_f * 0.01
        confidence_loss = torch.sum(confidence_loss) / batch_size_f # K.sum(confidence_loss) / batch_size_f
        class_loss = torch.sum(class_loss) / batch_size_f #K.sum(class_loss) / batch_size_f

        loss += location_loss + confidence_loss + class_loss + dist_loss

        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss
        total_dist_loss += dist_loss

    # Fit for tf 2.0.0 loss shape
    loss = loss.unsqueeze(-1) #K.expand_dims(loss, axis=-1)

    return loss, total_location_loss, total_confidence_loss, total_class_loss, total_dist_loss

## ----------------------------------------
##  Everything below is tensorflow code from Dist-YOLO which was converted above.
## Delete later (kept for reference)
## ----------------------------------------

# def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
#     """
#     Compute softmax focal loss.
#     Reference Paper:
#         "Focal Loss for Dense Object Detection"
#         https://arxiv.org/abs/1708.02002

#     # Arguments
#         y_true: Ground truth targets,
#             tensor of shape (?, num_boxes, num_classes).
#         y_pred: Predicted logits,
#             tensor of shape (?, num_boxes, num_classes).
#         gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
#         alpha: optional alpha weighting factor to balance positives vs negatives.

#     # Returns
#         softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
#     """

#     # Scale predictions so that the class probas of each sample sum to 1
#     #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

#     # Clip the prediction value to prevent NaN's and Inf's
#     #epsilon = K.epsilon()
#     #y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
#     y_pred = tf.nn.softmax(y_pred)
#     y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

#     # Calculate Cross Entropy
#     cross_entropy = -y_true * tf.math.log(y_pred)

#     # Calculate Focal Loss
#     softmax_focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

#     return softmax_focal_loss


# def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
#     """
#     Compute sigmoid focal loss.
#     Reference Paper:
#         "Focal Loss for Dense Object Detection"
#         https://arxiv.org/abs/1708.02002

#     # Arguments
#         y_true: Ground truth targets,
#             tensor of shape (?, num_boxes, num_classes).
#         y_pred: Predicted logits,
#             tensor of shape (?, num_boxes, num_classes).
#         gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
#         alpha: optional alpha weighting factor to balance positives vs negatives.

#     # Returns
#         sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
#     """
#     sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

#     pred_prob = tf.sigmoid(y_pred)
#     p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
#     modulating_factor = tf.pow(1.0 - p_t, gamma)
#     alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

#     sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
#     #sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

#     return sigmoid_focal_loss

# def box_iou(b1, b2):
#     """
#     Return iou tensor

#     Parameters
#     ----------
#     b1: tensor, shape=(i1,...,iN, 4), xywh
#     b2: tensor, shape=(j, 4), xywh

#     Returns
#     -------
#     iou: tensor, shape=(i1,...,iN, j)
#     """
#     # Expand dim to apply broadcasting.
#     b1 = K.expand_dims(b1, -2)
#     b1_xy = b1[..., :2]
#     b1_wh = b1[..., 2:4]
#     b1_wh_half = b1_wh/2.
#     b1_mins = b1_xy - b1_wh_half
#     b1_maxes = b1_xy + b1_wh_half

#     # Expand dim to apply broadcasting.
#     b2 = K.expand_dims(b2, 0)
#     b2_xy = b2[..., :2]
#     b2_wh = b2[..., 2:4]
#     b2_wh_half = b2_wh/2.
#     b2_mins = b2_xy - b2_wh_half
#     b2_maxes = b2_xy + b2_wh_half

#     intersect_mins = K.maximum(b1_mins, b2_mins)
#     intersect_maxes = K.minimum(b1_maxes, b2_maxes)
#     intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
#     intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#     b1_area = b1_wh[..., 0] * b1_wh[..., 1]
#     b2_area = b2_wh[..., 0] * b2_wh[..., 1]
#     iou = intersect_area / (b1_area + b2_area - intersect_area)

#     return iou


# def box_giou(b_true, b_pred):
#     """
#     Calculate GIoU loss on anchor boxes
#     Reference Paper:
#         "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
#         https://arxiv.org/abs/1902.09630

#     Parameters
#     ----------
#     b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
#     b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

#     Returns
#     -------
#     giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
#     """
#     b_true_xy = b_true[..., :2]
#     b_true_wh = b_true[..., 2:4]
#     b_true_wh_half = b_true_wh/2.
#     b_true_mins = b_true_xy - b_true_wh_half
#     b_true_maxes = b_true_xy + b_true_wh_half

#     b_pred_xy = b_pred[..., :2]
#     b_pred_wh = b_pred[..., 2:4]
#     b_pred_wh_half = b_pred_wh/2.
#     b_pred_mins = b_pred_xy - b_pred_wh_half
#     b_pred_maxes = b_pred_xy + b_pred_wh_half

#     intersect_mins = K.maximum(b_true_mins, b_pred_mins)
#     intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
#     intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
#     intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#     b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
#     b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
#     union_area = b_true_area + b_pred_area - intersect_area
#     # calculate IoU, add epsilon in denominator to avoid dividing by 0
#     iou = intersect_area / (union_area + K.epsilon())

#     # get enclosed area
#     enclose_mins = K.minimum(b_true_mins, b_pred_mins)
#     enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
#     enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
#     enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
#     # calculate GIoU, add epsilon in denominator to avoid dividing by 0
#     giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
#     giou = K.expand_dims(giou, -1)

#     return giou


# def box_diou(b_true, b_pred, use_ciou=False):
#     """
#     Calculate DIoU/CIoU loss on anchor boxes
#     Reference Paper:
#         "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
#         https://arxiv.org/abs/1911.08287

#     Parameters
#     ----------
#     b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
#     b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
#     use_ciou: bool flag to indicate whether to use CIoU loss type

#     Returns
#     -------
#     diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
#     """
#     b_true_xy = b_true[..., :2]
#     b_true_wh = b_true[..., 2:4]
#     b_true_wh_half = b_true_wh/2.
#     b_true_mins = b_true_xy - b_true_wh_half
#     b_true_maxes = b_true_xy + b_true_wh_half

#     b_pred_xy = b_pred[..., :2]
#     b_pred_wh = b_pred[..., 2:4]
#     b_pred_wh_half = b_pred_wh/2.
#     b_pred_mins = b_pred_xy - b_pred_wh_half
#     b_pred_maxes = b_pred_xy + b_pred_wh_half

#     intersect_mins = K.maximum(b_true_mins, b_pred_mins)
#     intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
#     intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
#     intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#     b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
#     b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
#     union_area = b_true_area + b_pred_area - intersect_area
#     # calculate IoU, add epsilon in denominator to avoid dividing by 0
#     iou = intersect_area / (union_area + K.epsilon())

#     # box center distance
#     center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
#     # get enclosed area
#     enclose_mins = K.minimum(b_true_mins, b_pred_mins)
#     enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
#     enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
#     # get enclosed diagonal distance
#     enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
#     # calculate DIoU, add epsilon in denominator to avoid dividing by 0
#     diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

#     if use_ciou:
#         # calculate param v and alpha to extend to CIoU
#         v = 4*K.square(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)

#         # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
#         #          to match related description for equation (12) in original paper
#         #
#         #
#         #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
#         #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
#         #
#         #          The dominator w^2+h^2 is usually a small value for the cases
#         #          h and w ranging in [0; 1], which is likely to yield gradient
#         #          explosion. And thus in our implementation, the dominator
#         #          w^2+h^2 is simply removed for stable convergence, by which
#         #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
#         #          is still consistent with Eqn. (12).
#         v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])

#         alpha = v / (1.0 - iou + v)
#         diou = diou - alpha*v

#     diou = K.expand_dims(diou, -1)
#     return diou


# def _smooth_labels(y_true, label_smoothing):
#     label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
#     return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


# def yolo3_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, elim_grid_sense=False, use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=True):
#     '''
#     YOLOv3 loss function.

#     Parameters
#     ----------
#     yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
#     y_true: list of array, the output of preprocess_true_boxes
#     anchors: array, shape=(N, 2), wh
#     num_classes: integer
#     ignore_thresh: float, the iou threshold whether to ignore object confidence loss

#     Returns
#     -------
#     loss: tensor, shape=(1,)

#     '''
#     num_layers = len(anchors)//3 # default setting
#     yolo_outputs = args[:num_layers]
#     y_true = args[num_layers:]

#     if num_layers == 3:
#         anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
#         scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
#     else:
#         anchor_mask = [[3,4,5], [0,1,2]]
#         scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

#     input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
#     grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
#     loss = 0
#     total_location_loss = 0
#     total_confidence_loss = 0
#     total_class_loss = 0
#     total_dist_loss = 0
#     batch_size = K.shape(yolo_outputs[0])[0] # batch size, tensor
#     batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

#     for l in range(num_layers):
#         object_mask = y_true[l][..., 4:5]
#         true_class_probs = y_true[l][..., 5:5+num_classes]
#         if label_smoothing:
#             true_class_probs = _smooth_labels(true_class_probs, label_smoothing)
#             true_objectness_probs = _smooth_labels(object_mask, label_smoothing)
#         else:
#             true_objectness_probs = object_mask

#         grid, raw_pred, pred_xy, pred_wh = yolo3_decode(yolo_outputs[l],
#              anchors[anchor_mask[l]], num_classes, input_shape, scale_x_y=scale_x_y[l], calc_loss=True)
#         pred_box = K.concatenate([pred_xy, pred_wh])

#         # Darknet raw box to calculate loss.
#         raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
#         raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
#         raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
#         box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

#         raw_true_dist = y_true[l][..., 5+num_classes:5+num_classes+1]
#         # Find ignore mask, iterate over each of batch.
#         ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
#         object_mask_bool = K.cast(object_mask, 'bool')
#         def loop_body(b, ignore_mask):
#             true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
#             iou = box_iou(pred_box[b], true_box)
#             best_iou = K.max(iou, axis=-1)
#             ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
#             return b+1, ignore_mask
#         _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
#         ignore_mask = ignore_mask.stack()
#         ignore_mask = K.expand_dims(ignore_mask, -1)

#         if use_focal_obj_loss:
#             # Focal loss for objectness confidence
#             confidence_loss = sigmoid_focal_loss(true_objectness_probs, raw_pred[...,4:5])
#         else:
#             confidence_loss = object_mask * K.binary_crossentropy(true_objectness_probs, raw_pred[...,4:5], from_logits=True)+ \
#                 (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask

#         if use_focal_loss:
#             # Focal loss for classification score
#             if use_softmax_loss:
#                 class_loss = softmax_focal_loss(true_class_probs, raw_pred[...,5:5+num_classes])
#             else:
#                 class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[...,5:5+num_classes])
#         else:
#             if use_softmax_loss:
#                 # use softmax style classification output
#                 class_loss = object_mask * K.expand_dims(K.categorical_crossentropy(true_class_probs, raw_pred[...,5:5+num_classes], from_logits=True), axis=-1)
#             else:
#                 # use sigmoid style classification output
#                 class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:5+num_classes], from_logits=True)


#         if use_giou_loss:
#             # Calculate GIoU loss as location loss
#             raw_true_box = y_true[l][...,0:4]
#             giou = box_giou(raw_true_box, pred_box)
#             giou_loss = object_mask * box_loss_scale * (1 - giou)
#             giou_loss = K.sum(giou_loss) / batch_size_f
#             location_loss = giou_loss
#         elif use_diou_loss:
#             # Calculate DIoU loss as location loss
#             raw_true_box = y_true[l][...,0:4]
#             diou = box_diou(raw_true_box, pred_box)
#             diou_loss = object_mask * box_loss_scale * (1 - diou)
#             diou_loss = K.sum(diou_loss) / batch_size_f
#             location_loss = diou_loss
#         else:
#             # Standard YOLOv3 location loss
#             # K.binary_crossentropy is helpful to avoid exp overflow.
#             xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
#             wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
#             xy_loss = K.sum(xy_loss) / batch_size_f
#             wh_loss = K.sum(wh_loss) / batch_size_f
#             location_loss = xy_loss + wh_loss

#         dist_loss = object_mask * 0.5 * K.square(raw_true_dist - raw_pred[..., 5+num_classes:5+num_classes+1])

#         dist_loss = K.sum(dist_loss) / batch_size_f * 0.01
#         confidence_loss = K.sum(confidence_loss) / batch_size_f
#         class_loss = K.sum(class_loss) / batch_size_f

#         loss += location_loss + confidence_loss + class_loss + dist_loss

#         total_location_loss += location_loss
#         total_confidence_loss += confidence_loss
#         total_class_loss += class_loss
#         total_dist_loss += dist_loss

#     # Fit for tf 2.0.0 loss shape
#     loss = K.expand_dims(loss, axis=-1)

#     return loss, total_location_loss, total_confidence_loss, total_class_loss, total_dist_loss

