import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from .utils import ann_xywh_to_xyxy
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


def target_class_loss_v1(cur_pred, init_pred, cls_logits, num_classes = 20, score_thr = 0.3):
    pred_bboxes = cur_pred.bboxes
    pred_labels = cur_pred.labels
    pred_scores = cur_pred.scores
    gt_bboxes = init_pred.bboxes
    gt_labels = init_pred.labels
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = 0
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
        max_overlap, argmax_overlaps = ious.max(1)
        if max_overlap < 0.01:
            continue
        # matched_bboxes = pred_bboxes[argmax_overlaps.item()]
        matched_logits = cls_logits[argmax_overlaps]
        # iou_loss += 1 - max_overlap + 1e-6
        # cls_loss += matched_logits[0][pred_labels[argmax_overlaps.item()]] - matched_logits[0][gt_label]
        cls_loss += ce_loss(matched_logits[:, :num_classes], gt_label[None])
    cls_loss /= gt_bboxes.shape[0]
    return cls_loss


def target_loss(cur_pred, init_pred, cls_logits, num_classes = 20, score_thr = 0.3):
    pred_bboxes = cur_pred.bboxes
    pred_labels = cur_pred.labels
    pred_scores = cur_pred.scores
    gt_bboxes = init_pred.bboxes
    gt_labels = init_pred.labels
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = 0
    iou_loss = 0
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
        max_overlap, argmax_overlaps = ious.max(1)
        if max_overlap < 0.01:
            continue
        matched_bboxes = pred_bboxes[argmax_overlaps.item()]
        matched_logits = cls_logits[argmax_overlaps]
        iou_loss += 1 - max_overlap + 1e-6
        # cls_loss += - matched_logits[0][gt_label]
        # alpha = (matched_logits[matched_logits > matched_logits[0][gt_label]].sum() // matched_logits[0][gt_label]).item()
        logit_sum = matched_logits[matched_logits > matched_logits[0][gt_label]].sum()
        cls_loss += logit_sum -  3 * matched_logits[0][gt_label]
        # cls_loss += matched_logits[0][pred_labels[argmax_overlaps.item()]] - matched_logits[0][gt_label]
        # cls_loss += ce_loss(matched_logits[:, :num_classes], gt_label[None])
    
    # cls_loss /= gt_bboxes.shape[0]
    # return iou_loss / gt_bboxes.shape[0]
    return (cls_loss + iou_loss) / gt_bboxes.shape[0]

# def target_feature_loss(cur_pred, init_pred, cls_logits, feats, model, dataset, num_classes = 20, score_thr = 0.3):
#     pred_bboxes = cur_pred.bboxes
#     pred_labels = cur_pred.labels
#     pred_scores = cur_pred.scores
#     gt_bboxes = init_pred.bboxes
#     gt_labels = init_pred.labels
#     ce_loss = nn.CrossEntropyLoss()
#     cls_loss = 0
#     iou_loss = 0
#     feat_loss = 0
#     for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
#         ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
#         max_overlap, argmax_overlaps = ious.max(1)
#         if max_overlap < 0.01:
#             continue
#         matched_bboxes = pred_bboxes[argmax_overlaps.item()]
#         matched_logits = cls_logits[argmax_overlaps]
#         feature_bbox = torch.round(matched_bboxes / 4).int()
#         bbox_feature = feats[..., feature_bbox[1]:feature_bbox[3], feature_bbox[0]:feature_bbox[2]]
#         support_img = random.choice(dataset.per_class_imgs[gt_label.item() + 1])
#         anns = dataset.per_img_anns[support_img]
#         target_anns = [ann for ann in anns if ann['category_id']==(gt_label.item() + 1)]
#         target_instance = random.choice(target_anns)['bbox']
#         target_instance = ann_xywh_to_xyxy(target_instance, 4)
#         support_img = dataset.data_prefix.img + support_img
#         sup_img = cv2.imread(support_img)
#         sup_data = {
#             "inputs" : [torch.tensor(sup_img).permute(2,0,1).cuda()],
#         }
#         sup_data = model.data_preprocessor(sup_data, False)
#         features = model._run_forward(sup_data, mode='feature')
#         target_bbox_feature = features[..., target_instance[1]:target_instance[3], 
#                                         target_instance[0]:target_instance[2]]
#         target_bbox_feature = F.interpolate(target_bbox_feature, bbox_feature.shape[2:], mode = 'bilinear', align_corners=False)
#         # feat_loss = F.kl_div(target_bbox_feature, bbox_feature, reduction = 'batchmean')
#         feat_loss = F.mse_loss(target_bbox_feature, bbox_feature)
#         iou_loss += 1 - max_overlap + 1e-6
#         # cls_loss += - matched_logits[0][gt_label]
#         # alpha = (matched_logits[matched_logits > matched_logits[0][gt_label]].sum() // matched_logits[0][gt_label]).item()
#         cls_loss += matched_logits[matched_logits > matched_logits[0][gt_label]].sum() - matched_logits[0][gt_label]
#         # cls_loss += matched_logits[0][pred_labels[argmax_overlaps.item()]] - matched_logits[0][gt_label]
#         # cls_loss += ce_loss(matched_logits[:, :num_classes], gt_label[None])
#         if cls_loss == 0:
#             return 0.0
#     # cls_loss /= gt_bboxes.shape[0]
#     # return iou_loss / gt_bboxes.shape[0]
#     return (cls_loss + iou_loss + feat_loss) / gt_bboxes.shape[0]

def target_feature_loss(cur_pred, init_pred, cls_logits, feats, target_feature, num_classes = 20, score_thr = 0.3):
    pred_bboxes = cur_pred.bboxes
    pred_labels = cur_pred.labels
    pred_scores = cur_pred.scores
    gt_bboxes = init_pred.bboxes
    gt_labels = init_pred.labels
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = 0
    iou_loss = 0
    feat_loss = 0
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
        max_overlap, argmax_overlaps = ious.max(1)
        if max_overlap < 0.01:
            continue
        matched_bboxes = pred_bboxes[argmax_overlaps.item()]
        matched_logits = cls_logits[argmax_overlaps]
        feature_bbox = torch.round(matched_bboxes / 4).int()
        bbox_feature = feats[..., feature_bbox[1]:feature_bbox[3], feature_bbox[0]:feature_bbox[2]]
        target_bbox_feature = target_feature[gt_label.item()]
        # target_bbox_feature = F.interpolate(target_bbox_feature, bbox_feature.shape[2:], mode = 'bilinear', align_corners=False)
        bbox_feature = F.interpolate(bbox_feature, target_bbox_feature.shape[2:], mode = 'bilinear', align_corners=False)
        N, C, H, W = bbox_feature.shape
        feat_loss = F.kl_div(target_bbox_feature.reshape(C, -1),
                             bbox_feature.reshape(C, -1), 
                             reduction = 'batchmean') / (H*W)
        # feat_loss = F.mse_loss(target_bbox_feature, bbox_feature)
        iou_loss += 1 - max_overlap + 1e-6
        # cls_loss += - matched_logits[0][gt_label]
        # alpha = (matched_logits[matched_logits > matched_logits[0][gt_label]].sum() // matched_logits[0][gt_label]).item()
        cls_loss += matched_logits[matched_logits > matched_logits[0][gt_label]].sum() - 5 * matched_logits[0][gt_label]
        # cls_loss += matched_logits[0][pred_labels[argmax_overlaps.item()]] - matched_logits[0][gt_label]
        # cls_loss += ce_loss(matched_logits[:, :num_classes], gt_label[None])
        if cls_loss == 0:
            print("cls loss == 0!")
            return 0.0
    # cls_loss /= gt_bboxes.shape[0]
    return (cls_loss + iou_loss + feat_loss)# / gt_bboxes.shape[0]


def target_loss_v1(cur_pred, init_pred, cls_logits, feats, target_feature, num_classes = 20, score_thr = 0.3):
    pred_bboxes = cur_pred.bboxes
    pred_labels = cur_pred.labels
    pred_scores = cur_pred.scores
    gt_bboxes = init_pred.bboxes
    gt_labels = init_pred.labels
    ce_loss = nn.CrossEntropyLoss()
    cls_loss = 0
    iou_loss = 0
    # feat_loss = 0
    for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
        ious = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
        matched_ious = ious[ious > 0.3]
        matched_bboxes = pred_bboxes[ious[0] > 0.3]
        matched_logits = cls_logits[ious[0] > 0.3]
        iou_loss += len(matched_ious) - matched_ious.sum()
        # n = 0.0
        for matched_logit in matched_logits:
            cls_loss += matched_logit[matched_logit > matched_logit[gt_label]].sum() - matched_logit[gt_label]
            # n += (matched_logit > matched_logit[gt_label]).sum()
        # if n > 0:
        #     cls_loss /= n
    return (cls_loss + iou_loss)# / gt_bboxes.shape[0]


def target_loss_v2(preds, init_pred, feats=None,num_classes = 20, score_thr = 0.3):
    tar_bboxes = init_pred.bboxes
    tar_labels = init_pred.labels
    cls_loss = 0
    iou_loss = 0 
    for tar_bbox, tar_label, pred in zip(tar_bboxes, tar_labels, preds):
        # ious = bbox_overlaps(tar_bbox.unsqueeze(0), pred.bboxes)[0]
        # matched_ious = ious[ious > 0.3]
        # matched_bboxes = pred.bboxes[ious[0] > 0.3]
        # matched_logits = pred.logits[ious[0] > 0.3]
        iou_loss += len(pred.iou) - pred.iou.sum()
        for logit in pred.logits:
            cls_loss += logit[logit > logit[tar_label]].sum() - logit[tar_label]
    return (cls_loss + iou_loss)



def feature_loss(feat, init_dets, target_label, model, dataset):
    feature_bboxes = init_dets.bboxes // 4
    for bbox in feature_bboxes:
        bbox_feature = feature_bboxes[bbox]
    
    


def faster_loss(cur_pred, init_pred, cls_logits, score_thr = 0.3):
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

    return loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from mmdet.utils import get_root_logger
# from functools import partial

# from ..builder import LOSSES


# @LOSSES.register_module()
# class EQLv2(nn.Module):
#     def __init__(self,
#                  use_sigmoid=True,
#                  reduction='mean',
#                  class_weight=None,
#                  loss_weight=1.0,
#                  num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
#                  gamma=12,
#                  mu=0.8,
#                  alpha=4.0,
#                  vis_grad=False,
#                  test_with_obj=True):
#         super().__init__()
#         self.use_sigmoid = True
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.class_weight = class_weight
#         self.num_classes = num_classes
#         self.group = True

#         # cfg for eqlv2
#         self.vis_grad = vis_grad
#         self.gamma = gamma
#         self.mu = mu
#         self.alpha = alpha

#         # initial variables
#         self.register_buffer('pos_grad', torch.zeros(self.num_classes))
#         self.register_buffer('neg_grad', torch.zeros(self.num_classes))
#         # At the beginning of training, we set a high value (eg. 100)
#         # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
#         self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

#         self.test_with_obj = test_with_obj

#         def _func(x, gamma, mu):
#             return 1 / (1 + torch.exp(-gamma * (x - mu)))
#         self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
#         logger = get_root_logger()
#         logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         self.n_i, self.n_c = cls_score.size()

#         self.gt_classes = label
#         self.pred_class_logits = cls_score

#         def expand_label(pred, gt_classes):
#             target = pred.new_zeros(self.n_i, self.n_c)
#             target[torch.arange(self.n_i), gt_classes] = 1
#             return target

#         target = expand_label(cls_score, label)

#         pos_w, neg_w = self.get_weight(cls_score)

#         weight = pos_w * target + neg_w * (1 - target)

#         cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
#                                                       reduction='none')
#         cls_loss = torch.sum(cls_loss * weight) / self.n_i

#         self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

#         return self.loss_weight * cls_loss

#     def collect_grad(self, cls_score, target, weight):
#         prob = torch.sigmoid(cls_score)
#         grad = target * (prob - 1) + (1 - target) * prob
#         grad = torch.abs(grad)

#         # do not collect grad for objectiveness branch [:-1]
#         pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
#         neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

#         dist.all_reduce(pos_grad)
#         dist.all_reduce(neg_grad)

#         self.pos_grad += pos_grad
#         self.neg_grad += neg_grad
#         self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

#     def get_weight(self, cls_score):
#         neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
#         pos_w = 1 + self.alpha * (1 - neg_w)
#         neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
#         pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
#         return pos_w, neg_w