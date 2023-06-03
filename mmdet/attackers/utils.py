import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as st
import copy
from mmengine.structures.instance_data import InstanceData
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps

def calc_iou(bbox1,bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
    
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def mask_img_result_change(same_class_box,same_class_pred,class_name,boxes_mask , labels_mask):
    if class_name not in labels_mask:
        det_pre = np.ones((len(same_class_pred),),dtype=np.float32)
        max_iou = np.zeros((len(same_class_pred),),dtype=np.float32)
        det_score = 1-max_iou+det_pre
    else:  
        same_class_box_mask = boxes_mask[labels_mask==class_name]
        bbox1 = np.array([box[0] for box in same_class_box])
        bbox2 = same_class_box_mask[:,:4]
        boxes_iou = calc_iou(bbox1,bbox2)
        max_iou = np.max(boxes_iou,axis=1)   
        max_index = np.argmax(boxes_iou,axis=1)
        max_pred = same_class_box_mask[:,4][max_index]
        det_pre = same_class_pred-max_pred
        det_score = 1-max_iou+det_pre
    
    return det_pre, max_iou, det_score

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def region_img_mean(img_cv2,x1,y1,w,h,i,j,divide_number):
    img_mean = img_cv2.squeeze().float()
    img_mean_0 = torch.mean(img_mean[0,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    img_mean_1 = torch.mean(img_mean[1,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    img_mean_2 = torch.mean(img_mean[2,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)])
    mn = torch.tensor([img_mean_0.item(),img_mean_1.item(),img_mean_2.item()])
    return mn



def get_valid_det():
    pass

def get_bbox_and_score(inds, num_classes, bboxes, scores):
    scores_ = scores[inds // num_classes]
    # bboxes = bboxes[inds // num_classes]
    # bboxes_ = []
    # for i, box in enumerate(bboxes):
    #     bboxes_.append(box[(inds[i]%20)*4:(inds[i]+1)%20*4])
    # bboxes_ = torch.stack(bboxes_)
    return scores_#, bboxes_

def get_target_label(logits, rank):
    labels = []
    for logit in logits:
        logit_list = logit[:20].tolist()
        labels.append(logit_list.index(sorted(logit_list)[rank - 1]))
    return torch.as_tensor(labels)


def get_target_label_v1(gts, dts, logits, rank):
    visited = [False] * len(dts)
    labels = []
    bboxes = []
    tar_ins = InstanceData()
    for gt in gts:
        ious = bbox_overlaps(gt.bboxes.tensor.cuda(), dts.bboxes)
        max_overlap, argmax_overlaps = ious.max(1)
        if max_overlap.item() > 0.5 and dts.scores[argmax_overlaps.item()] > 0.3:
            visited[argmax_overlaps.item()] = True
            logit_list = logits[argmax_overlaps.item()][:20].tolist()
            labels.append(torch.tensor([logit_list.index(sorted(logit_list)[rank - 1])]))
            bboxes.append(dts.bboxes[argmax_overlaps.item()])
    if bboxes == []:
        return None
    tar_ins.bboxes = torch.stack(bboxes).cuda()
    tar_ins.labels = torch.cat(labels).cuda()
    return tar_ins 
            

        
        
    labels = []
    for logit in logits:
        logit_list = logit[:20].tolist()
        labels.append(logit_list.index(sorted(logit_list)[rank - 1]))
    return torch.as_tensor(labels)
 

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

kernel = gkern().astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel])
stack_kernel = np.expand_dims(stack_kernel, 1)
stack_kernel = torch.from_numpy(stack_kernel).cuda()


def MI(grad_now, grad_pre):
    grad_norm = torch.norm(grad_now, p=1)
    grad = grad_now / grad_norm + grad_pre
    grad_pre = grad
    return grad, grad_pre
    
def BI(grad_now, grad_pre):
    return grad_now, grad_now


def TI(grad_now, grad_pre, kernel = stack_kernel):
    grad = F.conv2d(grad_now[None], kernel, padding=7, groups=3)
    return grad, grad_pre
    

def NI(grad_now, grad_pre):
    grad_norm = torch.norm(grad_now, p=1)
    grad = grad_now / grad_norm + grad_pre
    grad_pre = grad
    return grad, grad_pre