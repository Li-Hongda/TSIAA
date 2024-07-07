import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as st
import random
from mmengine.structures.instance_data import InstanceData
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps

def get_target_instance(gts, dts, logits, num_classes, rank_type):
    visited = [False] * len(dts)
    labels = []
    bboxes = []
    gts = gts.to(dts.bboxes.device)
    tar_ins = InstanceData()
    possible_ranks = [i for i in range(num_classes)]
    # np.random.seed(0)
    ranks = []
    for gt in gts:
        ious = bbox_overlaps(gt.bboxes.tensor.cuda(), dts.bboxes)[0]
        matched_idx = torch.nonzero(ious > 0.5)[:, 0]
        sorted_ious, sorted_index = ious[matched_idx].sort(descending=True)
        sorted_index = matched_idx[sorted_index]
        for i, idx in enumerate(sorted_index):
            if sorted_ious[i] > 0.5 and not visited[idx] and dts.labels[idx] == gt.labels:
                visited[idx] = True
                sorted_logits, sorted_indices = logits[idx][:num_classes].sort(descending=True)
                if rank_type == 'random':
                    # np.random.seed(0)
                    rank = np.random.randint(0, num_classes)
                    while rank == dts.labels[idx]:
                        rank = np.random.randint(0, num_classes)
                    ranks.append(rank)
                elif rank_type == 'worst':
                    rank = num_classes - 1
                else:
                    rank = rank_type
                labels.append(sorted_indices[rank-1])
                bboxes.append(gt.bboxes.tensor[0])
                break
    # print(ranks)
    if bboxes == []:
        return None
    tar_ins.bboxes = torch.stack(bboxes)
    tar_ins.labels = torch.stack(labels)
    return tar_ins 