import torch
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


def class_loss(cur_pred, init_pred, score_thr = 0.3):
    cls_score = cur_pred.scores
    mseloss = torch.nn.MSELoss()
    if cls_score[cls_score>=score_thr].shape[0]==0:
        class_loss=mseloss(cls_score*0, torch.zeros_like(cls_score).cuda())  #########
    else:
        class_loss = mseloss(cls_score, torch.zeros(cls_score.shape).cuda())  #########
    return class_loss

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