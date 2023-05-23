import numpy as np
import torch
import copy

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

def get_bbox_and_label():
    pass