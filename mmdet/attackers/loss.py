import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps

def assign_bboxes(pred_bboxes, init_bboxes, iou_thr):
    overlaps = bbox_overlaps(pred_bboxes, init_bboxes)
    init = []
    pred = []
    for i in range(len(overlaps)):
        overlap = overlaps[i]
        
        # _, argmax_overlaps = overlap.max(dim=0)
        # init.append(init_bboxes[argmax_overlaps].unsqueeze(0))
        # pred.append(pred_bboxes[i].unsqueeze(0))
        init.append(init_bboxes[overlap > iou_thr])
        pred.append(pred_bboxes[i].unsqueeze(0).repeat(init_bboxes[overlap > iou_thr].shape[0], 1))
    assigned_pred = torch.cat(pred)
    assigned_bboxes = torch.cat(init)
        
    return assigned_pred, assigned_bboxes


def class_loss(cur_pred, init_pred, cls_logits, score_thr = 0.3):
    cls_score = cur_pred.scores
    mseloss = torch.nn.MSELoss()
    if cls_score[cls_score>=score_thr].shape[0]==0:
        class_loss=mseloss(cls_score*0, torch.zeros_like(cls_score).cuda())  #########
    else:
        class_loss = mseloss(cls_score, torch.zeros(cls_score.shape).cuda())  #########
    return class_loss

def target_class_loss(cur_pred, init_pred, cls_logits, num_classes = 20, score_thr = 0.3):
    pred_bboxes = cur_pred.bboxes
    pred_labels = cur_pred.labels
    pred_scores = cur_pred.scores
    gt_bboxes = init_pred.bboxes
    gt_labels = init_pred.labels
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = 0
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes).squeeze()
        matched_bboxes = pred_bboxes[ious > 0]
        matched_logits = cls_logits[ious > 0]
        if matched_logits.shape[0] == 0:
            continue        
        if len(matched_logits.shape) > 2:
            matched_logits = matched_logits[0]
        onehot_label = F.one_hot(gt_label, num_classes).cuda()
        cls_loss += ce_loss(matched_logits[:,:num_classes], gt_label.repeat(matched_logits.shape[0]).cuda())            
    cls_loss /= gt_bboxes.shape[0]
    return cls_loss


def faster_loss(cur_pred, init_pred, score_thr = 0.3):
    cls_score = cur_pred.scores
    mseloss = torch.nn.MSELoss()
    if cls_score[cls_score>=score_thr].shape[0]==0:
        class_loss=mseloss(cls_score*0, torch.zeros_like(cls_score).cuda())  #########
        iou_loss = torch.zeros([1]).cuda()
    else:
        class_loss = mseloss(cls_score, torch.zeros(cls_score.shape).cuda())  #########
        pred_iou = bbox_overlaps(cur_pred.bboxes, init_pred.bboxes)
        # assigned_pred, assigned_bboxes = assign_bboxes(cur_pred.bboxes, init_pred.bboxes, 0)
        # pred_iou = bbox_overlaps(assigned_pred, assigned_bboxes)
        iou_loss = torch.sum(pred_iou)/cur_pred.labels.shape[0]
    loss = class_loss + iou_loss

    return loss, class_loss, iou_loss