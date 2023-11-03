# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool

import torch
import numpy as np
from mmengine import track_iter_progress
from mmengine.logging import print_log
from mmengine.utils import is_str
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps
from .class_names import get_classes


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None,
                 use_legacy_coordinate=False,
                 **kwargs):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Defaults to None
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be
            evaluated, in the format [(min1, max1), (min2, max2), ...].
            Defaults to None.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (
                det_bboxes[:, 2] - det_bboxes[:, 0] + extra_length) * (
                    det_bboxes[:, 3] - det_bboxes[:, 1] + extra_length)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps(
        det_bboxes, gt_bboxes, use_legacy_coordinate=use_legacy_coordinate)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + extra_length) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + extra_length)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + extra_length) * (
                    bbox[3] - bbox[1] + extra_length)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def get_cls_group_ofs(annotations, class_id):
    """Get `gt_group_of` of a certain class, which is used in Open Images.

    Args:
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        list[np.ndarray]: `gt_group_of` of a certain class.
    """
    gt_group_ofs = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        if ann.get('gt_is_group_ofs', None) is not None:
            gt_group_ofs.append(ann['gt_is_group_ofs'][gt_inds])
        else:
            gt_group_ofs.append(np.empty((0, 1), dtype=bool))

    return gt_group_ofs


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             ioa_thr=None,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4,
             use_legacy_coordinate=False,
             use_group_of=False,
             eval_mode='area'):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Defaults to None.
        iou_thr (float): IoU threshold to be considered as matched.
            Defaults to 0.5.
        ioa_thr (float | None): IoA threshold to be considered as matched,
            which only used in OpenImages evaluation. Defaults to None.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc", "imagenet_det", etc. Defaults to None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmengine.logging.print_log()` for details.
            Defaults to None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Defaults to 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Defaults to False.
        use_group_of (bool): Whether to use group of when calculate TP and FP,
            which only used in OpenImages evaluation. Defaults to False.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1],
            PASCAL VOC2007 uses `11points` as default evaluate mode, while
            others are 'area'. Defaults to 'area'.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    assert eval_mode in ['area', '11points'], \
        f'Unrecognized {eval_mode} mode, only "area" and "11points" ' \
        'are supported'
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # There is no need to use multi processes to process
    # when num_imgs = 1 .
    if num_imgs > 1:
        assert nproc > 0, 'nproc must be at least one.'
        nproc = min(nproc, num_imgs)
        pool = Pool(nproc)

    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        if num_imgs > 1:
            # compute tp and fp for each image with multiple processes
            args = []
            if use_group_of:
                # used in Open Images Dataset evaluation
                gt_group_ofs = get_cls_group_ofs(annotations, i)
                args.append(gt_group_ofs)
                args.append([use_group_of for _ in range(num_imgs)])
            if ioa_thr is not None:
                args.append([ioa_thr for _ in range(num_imgs)])

            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_ignore,
                    [iou_thr for _ in range(num_imgs)],
                    [area_ranges for _ in range(num_imgs)],
                    [use_legacy_coordinate for _ in range(num_imgs)], *args))
        else:
            tpfp = tpfp_fn(
                cls_dets[0],
                cls_gts[0],
                cls_gts_ignore[0],
                iou_thr,
                area_ranges,
                use_legacy_coordinate,
                gt_bboxes_group_of=(get_cls_group_ofs(annotations, i)[0]
                                    if use_group_of else None),
                use_group_of=use_group_of,
                ioa_thr=ioa_thr)
            tpfp = [tpfp]

        if use_group_of:
            tp, fp, cls_dets = tuple(zip(*tpfp))
        else:
            tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + extra_length) * (
                    bbox[:, 3] - bbox[:, 1] + extra_length)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        ap = average_precision(recalls, precisions, eval_mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })

    if num_imgs > 1:
        pool.close()

    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmengine.logging.print_log()` for details.
            Defaults to None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    elif is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        
        
def eval_asr(results, iou_thr=0.5):
    fp = 0
    tar = 0
    total = 0
    from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
    for result in track_iter_progress(results):
        annotation, det_result = result
        gt_bboxes = annotation['bboxes']
        gt_labels = annotation['labels']
        pred_scores = det_result['scores']
        pred_bboxes = det_result['bboxes'][pred_scores > 0.5]
        pred_labels = det_result['labels'][pred_scores > 0.5]
        total += gt_bboxes.shape[0]
        visited = [False] * gt_bboxes.shape[0]
        # TODO:一个预测框对应多个gt时如何计算？
        for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
            iou = bbox_overlaps(pred_bbox.unsqueeze(0), gt_bboxes)
            max_overlap, argmax_overlaps = iou.max(1)
            if max_overlap > iou_thr:
                # match_bbox = gt_bboxes[a rgmax_overlaps]
                sorted_idx = torch.argsort(iou[0],descending=True)[:len(iou[iou>0])]
                for idx in sorted_idx:
                    if pred_label == gt_labels[idx] and visited[idx] == False:
                        tar += 1
                        visited[idx] = True
                        break
                # match_label = gt_labels[argmax_overlaps]
                # # total += 1
                # if pred_label == match_label and visited[argmax_overlaps] == False:
                #     tar += 1
                # visited[argmax_overlaps] = True
            else:
                fp += 1         
    asr = tar / total * 100
    fr = fp / total * 100
    return asr, fr


def eval_dr(results, iou_thr=0.5):
    tar = 0
    total = 0
    from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
    for result in track_iter_progress(results):
        annotation, det_result = result
        gt_bboxes = annotation['bboxes']
        # gt_labels = annotation['labels']
        pred_scores = det_result['scores']
        pred_bboxes = det_result['bboxes'][pred_scores > 0.3]
        if pred_bboxes.shape[0] == 0:
            tar += gt_bboxes.shape[0]
            total += gt_bboxes.shape[0]
            continue
        for gt_bbox in gt_bboxes:
            iou = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
            max_overlap, _ = iou.max(1)
            total += 1
            if max_overlap <= iou_thr:
                tar += 1
    dr = tar / total * 100
    return dr

def eval_dr_v1(results, iou_thr=0):
    disappear = 0
    targeted = 0
    total = 0
    from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
    for result in track_iter_progress(results):
        annotation, det_result = result
        gt_bboxes = annotation['bboxes']
        gt_labels = annotation['labels']
        pred_scores = det_result['scores']
        pred_bboxes = det_result['bboxes'][pred_scores > 0.3]
        pred_labels = det_result['labels'][pred_scores > 0.3]
        if pred_bboxes.shape[0] == 0:
            disappear += gt_bboxes.shape[0]
            total += gt_bboxes.shape[0]
            continue
        visited = [False] * gt_bboxes.shape[0]
        for idx, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
            iou = bbox_overlaps(gt_bbox.unsqueeze(0), pred_bboxes)
            max_overlap, argmax_overlaps = iou.max(1)
            total += 1
            if max_overlap <= iou_thr:
                disappear += 1
            elif gt_label  == pred_labels[argmax_overlaps] and visited[idx] == False:
                targeted += 1
                visited[idx] = True
                
    dr = disappear / total * 100
    asr = targeted / total * 100
    return asr, dr

def eval_asr_v1(results, iou_thr=0):
    fp = 0
    total = 0
    from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
    for result in track_iter_progress(results):
        annotation, det_result = result
        gt_bboxes = annotation['bboxes']
        pred_scores = det_result['scores']
        pred_bboxes = det_result['bboxes'][pred_scores > 0.3]
        total += pred_bboxes.shape[0]
        for pred_bbox in pred_bboxes:
            iou = bbox_overlaps(pred_bbox.unsqueeze(0), gt_bboxes)
            max_overlap, argmax_overlaps = iou.max(1)
            if max_overlap <= iou_thr:
                fp += 1
    fr = fp / total * 100
    return fr


def print_summary(asr,
                  dr, 
                  fp,
                  logger=None):

    if logger == 'silent':
        return

    header = ['Attack Success Rate', 'Disappear Rate', 'FalsePositive Rate']

    table_data = [header]

    row_data = [
        f'{asr:.3f}', f'{dr:.3f}', f'{fp:.3f}'
    ]
    table_data.append(row_data)
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

