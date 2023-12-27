import torch.nn.functional as F
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.models.losses import weight_reduce_loss

def target_loss_v2(preds, init_pred, num_classes = 20):
    tar_bboxes = init_pred.bboxes
    tar_labels = init_pred.labels
    cls_loss = 0
    iou_loss = 0 
    for tar_bbox, tar_label, pred in zip(tar_bboxes, tar_labels, preds):
        if pred.labels.numel() == 0:
            continue
        iou_loss += len(pred.iou) - pred.iou.sum()
        cls_loss += F.cross_entropy(pred.logits, \
            tar_label.repeat(pred.logits.shape[0]))
    return (cls_loss + iou_loss)
               

def target_loss_v3(preds, tar_ins):
    tar_bboxes = tar_ins.bboxes
    tar_labels = tar_ins.labels
    num = 0 
    loss = 0
    for tar_label, pred in zip(tar_labels, preds):
        if pred.labels.numel() == 0:
            continue
        num += 1
        single_loss = (1 - pred.iou) + F.cross_entropy(pred.logits, \
            tar_label.repeat(pred.logits.shape[0]), reduction='none')
        loss += weight_reduce_loss(single_loss, \
            F.softmax(1 / pred.weights, dim=-1), reduction='mean')
    return loss / num
