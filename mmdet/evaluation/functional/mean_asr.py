# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool

import torch
import numpy as np
from mmengine import track_iter_progress
from mmengine.logging import print_log
from terminaltables import AsciiTable

from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
  
def eval_asr(results, iou_thr=0.5):
    fp = 0
    tar = 0
    total = 0
    total_pred = 0
    
    for result in track_iter_progress(results):
        annotation, det_result = result
        gt_bboxes = annotation['bboxes']
        gt_labels = annotation['labels']
        pred_scores = det_result['scores']
        pred_bboxes = det_result['bboxes'][pred_scores > 0]
        pred_labels = det_result['labels'][pred_scores > 0]
        total += gt_bboxes.shape[0]
        visited = [False] * gt_bboxes.shape[0]
        for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
            iou = bbox_overlaps(pred_bbox.unsqueeze(0), gt_bboxes)
            max_overlap, argmax_overlaps = iou.max(1)
            if max_overlap > iou_thr:
                sorted_idx = torch.argsort(iou[0],descending=True)[:len(iou[iou>iou_thr])]
                for idx in sorted_idx:
                    if pred_label == gt_labels[idx] and visited[idx] == False:
                        tar += 1
                        visited[idx] = True
                    elif pred_label == gt_labels[idx] and visited[idx] == True:
                        continue
                    elif pred_label != gt_labels[idx]:
                        fp += 1
                total_pred += len(sorted_idx)         
    asr = tar / total * 100
    fr = fp / total_pred * 100
    return asr, fr

def print_summary(asr,
                  fp,
                  logger=None):

    if logger == 'silent':
        return

    header = ['Attack Success Rate', 'FalsePositive Rate']

    table_data = [header]

    row_data = [
        f'{asr:.3f}', f'{fp:.3f}'
    ]
    table_data.append(row_data)
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    