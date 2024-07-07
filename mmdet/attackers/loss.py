import torch.nn.functional as F
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.models.losses import weight_reduce_loss
               

def target_loss(preds, tar_ins):
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
    return loss
