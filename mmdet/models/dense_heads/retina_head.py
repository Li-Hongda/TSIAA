# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList
from .anchor_head import AnchorHead
from ..utils import (filter_scores_and_topk, select_single_mlvl)

@MODELS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        assert stacked_convs >= 0, \
            '`stacked_convs` must be non-negative integers, ' \
            f'but got {stacked_convs} instead.'
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            in_channels = self.feat_channels
        self.retina_cls = nn.Conv2d(
            in_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors * reg_dim, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    # def predict_logits(self,
    #                     cls_scores: List[Tensor],
    #                     bbox_preds: List[Tensor],
    #                     score_factors: Optional[List[Tensor]] = None,
    #                     batch_img_metas: Optional[List[dict]] = None,
    #                     cfg: Optional[ConfigDict] = None,
    #                     rescale: bool = False,
    #                     with_nms: bool = True) -> InstanceList:
    #     """Transform a batch of output features extracted from the head into
    #     bbox results.

    #     Note: When score_factors is not None, the cls_scores are
    #     usually multiplied by it then obtain the real score used in NMS,
    #     such as CenterNess in FCOS, IoU branch in ATSS.

    #     Args:
    #         cls_scores (list[Tensor]): Classification scores for all
    #             scale levels, each is a 4D-tensor, has shape
    #             (batch_size, num_priors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for all
    #             scale levels, each is a 4D-tensor, has shape
    #             (batch_size, num_priors * 4, H, W).
    #         score_factors (list[Tensor], optional): Score factor for
    #             all scale level, each is a 4D-tensor, has shape
    #             (batch_size, num_priors * 1, H, W). Defaults to None.
    #         batch_img_metas (list[dict], Optional): Batch image meta info.
    #             Defaults to None.
    #         cfg (ConfigDict, optional): Test / postprocessing
    #             configuration, if None, test_cfg would be used.
    #             Defaults to None.
    #         rescale (bool): If True, return boxes in original image space.
    #             Defaults to False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Defaults to True.

    #     Returns:
    #         list[:obj:`InstanceData`]: Object detection results of each image
    #         after the post process. Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     assert len(cls_scores) == len(bbox_preds)

    #     if score_factors is None:
    #         # e.g. Retina, FreeAnchor, Foveabox, etc.
    #         with_score_factors = False
    #     else:
    #         # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
    #         with_score_factors = True
    #         assert len(cls_scores) == len(score_factors)

    #     num_levels = len(cls_scores)

    #     featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #     mlvl_priors = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=cls_scores[0].dtype,
    #         device=cls_scores[0].device)

    #     # result_list = []

    #     for img_id in range(len(batch_img_metas)):
    #         img_meta = batch_img_metas[img_id]
    #         cls_score_list = select_single_mlvl(
    #             cls_scores, img_id, detach=False)
    #         bbox_pred_list = select_single_mlvl(
    #             bbox_preds, img_id, detach=False)
    #         if with_score_factors:
    #             score_factor_list = select_single_mlvl(
    #                 score_factors, img_id, detach=False)
    #         else:
    #             score_factor_list = [None for _ in range(num_levels)]

    #         results, cls_logits = self.predict_logits_single(
    #             cls_score_list=cls_score_list,
    #             bbox_pred_list=bbox_pred_list,
    #             score_factor_list=score_factor_list,
    #             mlvl_priors=mlvl_priors,
    #             img_meta=img_meta,
    #             cfg=cfg,
    #             rescale=rescale,
    #             with_nms=with_nms)
    #         # result_list.append(results)
    #     return results, cls_logits

    # def predict_logits_single(self,
    #                             cls_score_list: List[Tensor],
    #                             bbox_pred_list: List[Tensor],
    #                             score_factor_list: List[Tensor],
    #                             mlvl_priors: List[Tensor],
    #                             img_meta: dict,
    #                             cfg: ConfigDict,
    #                             rescale: bool = False,
    #                             with_nms: bool = True) -> InstanceData:
    #     """Transform a single image's features extracted from the head into
    #     bbox results.

    #     Args:
    #         cls_score_list (list[Tensor]): Box scores from all scale
    #             levels of a single image, each item has shape
    #             (num_priors * num_classes, H, W).
    #         bbox_pred_list (list[Tensor]): Box energies / deltas from
    #             all scale levels of a single image, each item has shape
    #             (num_priors * 4, H, W).
    #         score_factor_list (list[Tensor]): Score factor from all scale
    #             levels of a single image, each item has shape
    #             (num_priors * 1, H, W).
    #         mlvl_priors (list[Tensor]): Each element in the list is
    #             the priors of a single level in feature pyramid. In all
    #             anchor-based methods, it has shape (num_priors, 4). In
    #             all anchor-free methods, it has shape (num_priors, 2)
    #             when `with_stride=True`, otherwise it still has shape
    #             (num_priors, 4).
    #         img_meta (dict): Image meta info.
    #         cfg (mmengine.Config): Test / postprocessing configuration,
    #             if None, test_cfg would be used.
    #         rescale (bool): If True, return boxes in original image space.
    #             Defaults to False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Defaults to True.

    #     Returns:
    #         :obj:`InstanceData`: Detection results of each image
    #         after the post process.
    #         Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     if score_factor_list[0] is None:
    #         # e.g. Retina, FreeAnchor, etc.
    #         with_score_factors = False
    #     else:
    #         # e.g. FCOS, PAA, ATSS, etc.
    #         with_score_factors = True

    #     cfg = self.test_cfg if cfg is None else cfg
    #     cfg = copy.deepcopy(cfg)
    #     img_shape = img_meta['img_shape']
    #     nms_pre = cfg.get('nms_pre', -1)

    #     mlvl_logits = []
    #     mlvl_bbox_preds = []
    #     mlvl_valid_priors = []
    #     mlvl_scores = []
    #     mlvl_labels = []
    #     if with_score_factors:
    #         mlvl_score_factors = []
    #     else:
    #         mlvl_score_factors = None
    #     for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
    #             enumerate(zip(cls_score_list, bbox_pred_list,
    #                           score_factor_list, mlvl_priors)):

    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

    #         dim = self.bbox_coder.encode_size
    #         bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
    #         if with_score_factors:
    #             score_factor = score_factor.permute(1, 2,
    #                                                 0).reshape(-1).sigmoid()
    #         cls_score = cls_score.permute(1, 2,
    #                                       0).reshape(-1, self.cls_out_channels)
    #         if self.use_sigmoid_cls:
    #             scores = cls_score.sigmoid()
    #         else:
    #             # remind that we set FG labels to [0, num_class-1]
    #             # since mmdet v2.0
    #             # BG cat_id: num_class
    #             scores = cls_score.softmax(-1)[:, :-1]

    #         # After https://github.com/open-mmlab/mmdetection/pull/6268/,
    #         # this operation keeps fewer bboxes under the same `nms_pre`.
    #         # There is no difference in performance for most models. If you
    #         # find a slight drop in performance, you can set a larger
    #         # `nms_pre` than before.
    #         score_thr = cfg.get('score_thr', 0)

    #         results = filter_scores_and_topk(
    #             scores, score_thr, nms_pre,
    #             dict(bbox_pred=bbox_pred, priors=priors))
    #         scores, labels, keep_idxs, filtered_results = results

    #         bbox_pred = filtered_results['bbox_pred']
    #         priors = filtered_results['priors']

    #         if with_score_factors:
    #             score_factor = score_factor[keep_idxs]

    #         mlvl_logits.append(cls_score[keep_idxs])
    #         mlvl_bbox_preds.append(bbox_pred)
    #         mlvl_valid_priors.append(priors)
    #         mlvl_scores.append(scores)
    #         mlvl_labels.append(labels)

    #         if with_score_factors:
    #             mlvl_score_factors.append(score_factor)

    #     bbox_pred = torch.cat(mlvl_bbox_preds)
    #     priors = cat_boxes(mlvl_valid_priors)
    #     bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

    #     results = InstanceData()
    #     results.bboxes = bboxes
    #     results.scores = torch.cat(mlvl_scores)
    #     results.labels = torch.cat(mlvl_labels)
    #     cls_logits = torch.cat(mlvl_logits)
    #     if with_score_factors:
    #         results.score_factors = torch.cat(mlvl_score_factors)
    #     results, cls_logits = self.post_process_logits(
    #         results=results,
    #         logits=cls_logits, 
    #         cfg=cfg,
    #         rescale=rescale,
    #         with_nms=with_nms,
    #         img_meta=img_meta)
    #     return results, cls_logits

    # def post_process_logits(self,
    #                        results: InstanceData,
    #                        logits: Tensor,
    #                        cfg: ConfigDict,
    #                        rescale: bool = False,
    #                        with_nms: bool = True,
    #                        img_meta: Optional[dict] = None) -> InstanceData:
    #     """bbox post-processing method.

    #     The boxes would be rescaled to the original image scale and do
    #     the nms operation. Usually `with_nms` is False is used for aug test.

    #     Args:
    #         results (:obj:`InstaceData`): Detection instance results,
    #             each item has shape (num_bboxes, ).
    #         cfg (ConfigDict): Test / postprocessing configuration,
    #             if None, test_cfg would be used.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default to False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default to True.
    #         img_meta (dict, optional): Image meta info. Defaults to None.

    #     Returns:
    #         :obj:`InstanceData`: Detection results of each image
    #         after the post process.
    #         Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     if rescale:
    #         assert img_meta.get('scale_factor') is not None
    #         scale_factor = [1 / s for s in img_meta['scale_factor']]
    #         results.bboxes = scale_boxes(results.bboxes, scale_factor)

    #     if hasattr(results, 'score_factors'):
    #         # TODO： Add sqrt operation in order to be consistent with
    #         #  the paper.
    #         score_factors = results.pop('score_factors')
    #         results.scores = results.scores * score_factors

    #     # filter small size bboxes
    #     if cfg.get('min_bbox_size', -1) >= 0:
    #         w, h = get_box_wh(results.bboxes)
    #         valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
    #         if not valid_mask.all():
    #             results = results[valid_mask]

    #     # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
    #     if with_nms and results.bboxes.numel() > 0:
    #         bboxes = get_box_tensor(results.bboxes)
    #         det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
    #                                             results.labels, cfg.nms)
    #         results = results[keep_idxs]
    #         # some nms would reweight the score, such as softnms
    #         results.scores = det_bboxes[:, -1]
    #         results = results[:cfg.max_per_img]

    #     return results, logits[keep_idxs][:cfg.max_per_img]