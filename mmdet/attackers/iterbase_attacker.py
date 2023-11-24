from abc import abstractmethod
import numpy as np
import copy
import cv2
from collections import OrderedDict
import torch
from mmengine.utils import is_list_of
from mmdet.registry import ATTACKERS
import torch.nn as nn
from .utils import *
from .loss import *


@ATTACKERS.register_module()
class IterBaseAttacker(nn.Module):    
    def __init__(self,
                 img_mean, 
                 img_std, 
                 steps = 10,
                 step_size = 1,
                 epsilon = 16,
                 targeted = False):
        
        self.eps = epsilon
        self.step_size = step_size
        self.steps = steps        
        self.mean = torch.tensor(img_mean, device = 'cuda')
        self.std = torch.tensor(img_std, device = 'cuda')
        self.targeted = targeted
        self.is_attack = True

    def prepare_model(self, model):
        for i, param in model.named_parameters():
            param.requires_grad = False         
        model.eval()
        return model
    
    @abstractmethod
    def attack(self, model, data, save = False):
        pass

    def normalize(self, img, mean, std):
        
        for i in range(3):
            img[:,i,:] = (img[:,i,:] - mean[i].cuda()) / std[i].cuda()
        return img

    def imnormalize(self, img, mean, std):
        for i in range(3):
            img[:,i,:] = img[:,i,:] * std[i].cuda() + mean[i].cuda()
        return img
    
    def get_init_results(self, data, model):
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
                num_classes = model.roi_head.bbox_head.num_classes
            else:
                init_results, cls_logits = model.bbox_head.predict_logits(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
                num_classes = model.bbox_head.num_classes
        return init_results, cls_logits, num_classes
    
    def attack_step(self, adv_images, model, loss_func, data, init_results, ori_images):
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        adv_data = {
            "inputs":[adv_images + delta],
            "data_samples":data['data_samples']
        }
        adv_data = model.data_preprocessor(adv_data, False)
        outs = model._run_forward(adv_data, mode='tensor')
        
        if hasattr(model, 'roi_head'):
            outs = [[out] for out in outs]
            results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
        else:
            results = model.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
        # if results[0].labels.shape[0] == 0: 
        #     break
        loss = loss_func(results[0], init_results[0], cls_logits)
        loss.backward()
        noise = delta.grad.data.detach_().clone()
        delta.grad.zero_()
        delta.data = delta.data + self.step_size * torch.sign(noise)
        delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
        delta.data = delta.data.clamp(-self.eps, self.eps)
        adv_images = ori_images.clone() + delta
        # del loss, delta
        return adv_images
        
    def make_init_mask_img(self, images, init_results, model, data):
        boxes_init,pred_init,labels_init = [],[],[]
        for i in range(len(init_results.labels)):
            if init_results.scores[i] > 0.3:
                boxes_init.append([int(init_results.bboxes[i][0]), int(init_results.bboxes[i][1]), 
                                   int(init_results.bboxes[i][2]), int(init_results.bboxes[i][3])])
                pred_init.append(init_results.scores[i].item())
                labels_init.append(init_results.labels[i].item()) 

        attack_map = np.zeros(images.shape[2:4])
        attack_map_mean = np.zeros(images.shape[2:4])
        attack_map_mean = np.stack((attack_map_mean, attack_map_mean, attack_map_mean),axis=-1)
        divide_size=[1,2,3]
        size_divide = {0:[],1:[],2:[]}
        for i in range(len(boxes_init)):
            area = np.abs(boxes_init[i][2]-boxes_init[i][0])*np.abs(boxes_init[i][3]-boxes_init[i][1])
            if area<1024:
                size_divide[0].append([boxes_init[i],pred_init[i],labels_init[i]])
            elif area >=1024 and area < 9216:
                size_divide[1].append([boxes_init[i],pred_init[i],labels_init[i]])
            else:
                size_divide[2].append([boxes_init[i],pred_init[i],labels_init[i]])
        for area_size,results in size_divide.items():
            if divide_size[area_size]==1:
                for box in results:
                    x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                    w = x2-x1
                    h = y2-y1
                    attack_map[y1:y2,x1:x2] = 1
                    img_mn = region_img_mean(images,x1,y1,w,h,0,0,1)
                    img_mean = img_mn.view(1,1,3)
                    attack_map_mean[y1:y2,x1:x2,:] = img_mean
            else:
                if results != []:
                    same_class = {}
                    for result in results:
                        if same_class.get(result[2]) == None:
                            same_class[result[2]] = [result]
                        else:
                            same_class[result[2]].append(result)
                    divide_number = divide_size[area_size]
                    for class_name, same_class_box in same_class.items():
                        all_iou = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        all_pred = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        all_score = {i:np.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        for i in range(divide_number):    
                            for j in range(divide_number):
                                same_class_pred = []
                                img_mask = copy.deepcopy(images)
                                for box in same_class_box:
                                    x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                                    w = x2-x1
                                    h = y2-y1
                                    same_class_pred.append(box[1])
                                    img_mn = region_img_mean(images,x1,y1,w,h,i,j,divide_number)
                                    img_mean = img_mn.view(1,3,1,1)
                                    img_mask[:,:,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)] = img_mean
                                with torch.no_grad():
                                    data['inputs'] = img_mask
                                    img = model.data_preprocessor(copy.deepcopy(data), False)
                                    outs = model._run_forward(img, mode='tensor')
                                    if hasattr(model, 'roi_head'):
                                        outs = [[out] for out in outs]
                                        results = model.roi_head.bbox_head.predict_by_feat(*outs, 
                                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
                                    else:
                                        results = model.bbox_head.predict_by_feat(*outs, 
                                                  batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)                            
                                boxes_mask = torch.cat((results[0].bboxes, results[0].scores.unsqueeze(1)),dim=-1).cpu().numpy()
                                labels_mask = results[0].labels.cpu().numpy()
                                # boxes_mask , labels_mask = faster_rcnn_attack_box(img_path, faster_rcnn_model, img_mask, img_mask)
                                det_pre, det_iou,det_score = mask_img_result_change(same_class_box,same_class_pred,class_name, boxes_mask , labels_mask)
                                for di in range(len(det_iou)):all_iou[di][i,j]=det_iou[di]
                                for dp in range(len(det_pre)):all_pred[dp][i,j]=det_pre[dp]
                                for ds in range(len(det_score)):all_score[ds][i,j]=(1-det_iou[ds])+det_pre[ds]
                                
                        save_pre = pow(divide_number,2)//2
                        # save_pre = 3
                        for ds in range(len(det_score)):
                            x1,y1,x2,y2 = same_class_box[ds][0][0],same_class_box[ds][0][1],same_class_box[ds][0][2],same_class_box[ds][0][3]
                            w = x2-x1
                            h = y2-y1
                            select_region = k_largest_index_argsort(all_score[ds],save_pre)
                            for sa in range(save_pre):
                                i,j=select_region[sa][0],select_region[sa][1]
                                attack_map[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)]=1
                                img_mn = region_img_mean(images,x1,y1,w,h,i,j,divide_number)
                                img_mean = img_mn.view(1,1,3)
                                attack_map_mean[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number),:]=img_mean
                            
        attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)       
        return attack_map,attack_map_mean


    def make_init_mask_img_torch(self, ori_images, init_results, model, data):
        pred_init = init_results.scores[init_results.scores > 0.3]
        boxes_init = init_results.bboxes[init_results.scores > 0.3]
        labels_init = init_results.labels[init_results.scores > 0.3]
        
        attack_map = torch.zeros(ori_images.shape[2:4])
        attack_map_mean = torch.zeros(ori_images.shape[2:4])
        attack_map_mean = torch.stack((attack_map_mean, attack_map_mean, attack_map_mean),axis=-1)
        divide_size=[1,2,3]
        size_divide = {0:[],1:[],2:[]}
        for i in range(len(boxes_init)):
            area = torch.abs(boxes_init[i][2]-boxes_init[i][0]) * torch.abs(boxes_init[i][3]-boxes_init[i][1])
            if area<1024:
                size_divide[0].append([boxes_init[i],pred_init[i],labels_init[i].item()])
            elif area >=1024 and area < 9216:
                size_divide[1].append([boxes_init[i],pred_init[i],labels_init[i].item()])
            else:
                size_divide[2].append([boxes_init[i],pred_init[i],labels_init[i].item()])
        for area_size,results in size_divide.items():
            if divide_size[area_size]==1:
                for box in results:
                    x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                    w = x2-x1
                    h = y2-y1
                    attack_map[y1:y2,x1:x2] = 1
                    img_mn = region_img_mean(ori_images,x1,y1,w,h,0,0,1)
                    img_mean = img_mn.view(1,1,3)
                    attack_map_mean[y1:y2,x1:x2,:] = img_mean
            else:
                if results != []:
                    same_class = {}
                    for result in results:
                        if same_class.get(result[2]) == None:
                            same_class[result[2]] = [result]
                        else:
                            same_class[result[2]].append(result)
                    divide_number = divide_size[area_size]
                    for class_name, same_class_box in same_class.items():
                        all_iou = {i:torch.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        all_pred = {i:torch.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        all_score = {i:torch.zeros((divide_number,divide_number)) for i in range(len(same_class_box))}
                        for i in range(divide_number):    
                            for j in range(divide_number):
                                same_class_pred = []
                                img_mask = copy.deepcopy(ori_images)
                                for box in same_class_box:
                                    x1,y1,x2,y2 = box[0][0],box[0][1],box[0][2],box[0][3]
                                    w = x2-x1
                                    h = y2-y1
                                    same_class_pred.append(box[1])
                                    img_mn = region_img_mean(ori_images,x1,y1,w,h,i,j,divide_number)
                                    img_mean = img_mn.view(1,3,1,1)
                                    img_mask[:,:,int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)] = img_mean
                            
                                with torch.no_grad():
                                    data['inputs'] = img_mask
                                    img = model.data_preprocessor(copy.deepcopy(data), False)
                                    outs = model._run_forward(img, mode='tensor')
                                    if hasattr(model, 'roi_head'):
                                        outs = [[out] for out in outs]
                                        results = model.roi_head.bbox_head.predict_by_feat(*outs, 
                                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
                                    else:
                                        results = model.bbox_head.predict_by_feat(*outs, 
                                                  batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)                            
                                boxes_mask = torch.cat((results[0].bboxes, results[0].scores.unsqueeze(1)),dim=-1).cpu().numpy()
                                labels_mask = results[0].labels.cpu().numpy()
                                # boxes_mask , labels_mask = faster_rcnn_attack_box(img_path, faster_rcnn_model, img_mask, img_mask)
                                det_pre, det_iou,det_score = mask_img_result_change(same_class_box,same_class_pred,class_name, boxes_mask , labels_mask)
                                for di in range(len(det_iou)):all_iou[di][i,j]=det_iou[di]
                                for dp in range(len(det_pre)):all_pred[dp][i,j]=det_pre[dp]
                                for ds in range(len(det_score)):all_score[ds][i,j]=(1-det_iou[ds])+det_pre[ds]
                                
                        save_pre = pow(divide_number,2)//2
                        # save_pre = 3
                        for ds in range(len(det_score)):
                            x1,y1,x2,y2 = same_class_box[ds][0][0],same_class_box[ds][0][1],same_class_box[ds][0][2],same_class_box[ds][0][3]
                            w = x2-x1
                            h = y2-y1
                            select_region = k_largest_index_argsort(all_score[ds],save_pre)
                            for sa in range(save_pre):
                                i,j=select_region[sa][0],select_region[sa][1]
                                attack_map[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number)]=1
                                img_mn = region_img_mean(ori_images,x1,y1,w,h,i,j,divide_number)
                                img_mean = img_mn.view(1,1,3)
                                attack_map_mean[int(y1+j*h/divide_number):int(y1+(j+1)*h/divide_number),int(x1+i*w/divide_number):int(x1+(i+1)*w/divide_number),:]=img_mean
                            
        attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)       
        return attack_map,attack_map_mean    

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        return repr_str    
    
    

@ATTACKERS.register_module()
class BIMAttacker(IterBaseAttacker):
    
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, _ = self.get_init_results(data, model)
        adv_images = ori_images.clone()
        for ii in range(self.steps):
            adv_images = self.attack_step(adv_images, model, class_loss, data, init_results, ori_images)
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        } 
        return adv_data
    

@ATTACKERS.register_module()
class BIMFODAttacker(IterBaseAttacker):
    
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, _ = self.get_init_results(data, model)
        attack_map, _ = self.make_init_mask_img(ori_images.unsqueeze(0), init_results[0], model, data)
        attack_map = torch.from_numpy(attack_map).permute(2,0,1).cuda()
        adv_images = ori_images.clone()
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True            
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, _ = model.roi_head.bbox_head.predict_by_feat(*outs, 
                          batch_img_metas=[data['data_samples'][0].metainfo], 
                          rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results = model.bbox_head.predict_by_feat(*outs, 
                          batch_img_metas=[data['data_samples'][0].metainfo], 
                          rescale=True)
            # if results[0].labels.shape[0] == 0:
            #     break
            loss= class_loss(results[0], init_results[0])
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            delta.grad.zero_()
            delta.data = delta.data + self.step_size * torch.sign(noise) * attack_map
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            adv_images = ori_images.clone() + delta
            del loss, delta
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }        
        return adv_data


@ATTACKERS.register_module()
class BIMIOUAttacker(BIMAttacker):
    
    def attack(self, model, data, save = False):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, _ = self.get_init_results(data, model)
        adv_images = ori_images.clone()
        for ii in range(self.steps):
            adv_images = self.attack_step(adv_images, model, faster_loss, data, init_results, ori_images)
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data


@ATTACKERS.register_module()
class BIMIOUFODAttacker(BIMAttacker):
    
    def attack(self, model, data, save = False):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, _ = self.get_init_results(data, model)
        attack_map, _ = self.make_init_mask_img(ori_images.unsqueeze(0), init_results[0], model, data)
        attack_map = torch.from_numpy(attack_map).permute(2,0,1).cuda()            
        adv_images = ori_images.clone()
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True            
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, _ = model.roi_head.bbox_head.predict_by_feat(*outs, 
                         batch_img_metas=[data['data_samples'][0].metainfo], 
                         rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results = model.bbox_head.predict_by_feat(*outs, 
                          batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            # if results[0].labels.shape[0] == 0:
            #     break
            loss, class_loss, iou_loss = faster_loss(results[0], init_results[0])
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            delta.grad.zero_()
            delta.data = delta.data + self.step_size * torch.sign(noise) * attack_map
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
            del loss, class_loss, iou_loss, delta
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data


@ATTACKERS.register_module()
class TBIMAttacker(BIMAttacker):
    def __init__(self, targeted = True, **kwargs): 
        super().__init__(**kwargs)
        self.targeted = targeted
    
    
    def attack_step(self, adv_images, model, loss_func, grad_func, data, init_results, ori_images, grad_pre, num_classes):
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        adv_data = {
            "inputs":[adv_images + delta],
            "data_samples":data['data_samples']
        }
        adv_data = model.data_preprocessor(adv_data, False)
        outs = model._run_forward(adv_data, mode='tensor')
        
        if hasattr(model, 'roi_head'):
            outs = [[out] for out in outs]
            results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
        else:
            results, cls_logits = model.bbox_head.inference(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
        if results.bboxes.shape[0] == 0:
            return adv_images, grad_pre
        loss = loss_func(results[0], init_results[0], cls_logits, num_classes)
        if loss == 0.0:
            return adv_images, grad_pre
        loss.backward()
        noise = delta.grad.data.detach_().clone()
        noise, grad_pre = grad_func(noise, grad_pre)
        if len(noise.shape) == 4:
            noise = noise[0]
        delta.grad.zero_()
        delta.data = delta.data - self.step_size * torch.sign(noise)
        delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
        delta.data = delta.data.clamp(-self.eps, self.eps)
        
        adv_images = ori_images.clone() + delta 
        return adv_images, grad_pre
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/20062.png':
        #     print() 
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)
        if init_results.bboxes.shape[0] == 0:
            return data

        tar_instances, ori_labels = get_target_instance_v0(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        # init_results.labels.data = tar_instances.labels

        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')           
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.predict_logits(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            loss = target_bim_loss_v1(results, tar_instances, ori_labels, cls_logits, num_classes)
            if loss == 0:
                # print("Attack failed.")
                adv_data = {
                    "inputs":[adv_images],
                    "data_samples":data['data_samples']
                }
                return adv_data
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            if len(noise.shape) == 4:
                noise = noise[0]
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta             
            # adv_images, grad_pre = self.attack_step(adv_images, model, target_loss, BI, data, init_results, ori_images, grad_pre, num_classes)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data


@ATTACKERS.register_module()
class TLBIMAttacker(BIMAttacker):
    def __init__(self, targeted = True, **kwargs): 
        super().__init__(**kwargs)
        self.targeted = targeted
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/20062.png':
        #     print() 
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)
        if init_results.bboxes.shape[0] == 0:
            return data

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        attack_map = self.get_attack_map(tar_instances, ori_images)
        data['data_samples'][0].gt_instances = tar_instances
        # init_results.labels.data = tar_instances.labels

        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')           
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.predict_logits(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            loss = target_bim_loss_v1(results, tar_instances, init_results, cls_logits, num_classes)
            if loss == 0:
                print("Attack failed.")
                adv_data = {
                    "inputs":[adv_images],
                    "data_samples":data['data_samples']
                }
                return adv_data
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            if len(noise.shape) == 4:
                noise = noise[0]
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise) * attack_map
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta             
            # adv_images, grad_pre = self.attack_step(adv_images, model, target_loss, BI, data, init_results, ori_images, grad_pre, num_classes)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    def get_attack_map(self, instances, images):
        attack_map = images.new_zeros(images.shape)
        for ins in instances:
            x1, y1, x2, y2 = list(map(int, ins.bboxes[0]))
            attack_map[:, y1:y2, x1:x2] = 1
        return attack_map    

@ATTACKERS.register_module()
class TFBIMAttacker(TBIMAttacker):

    # def attack_step(self, adv_images, model, loss_func, grad_func, data, init_results, ori_images, grad_pre):

    
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
        if tar_instances == None:
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            # print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        target_feature = self.get_target_feats(model, dataset, tar_instances)
        data['data_samples'][0].gt_instances = tar_instances
        init_results[0].labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            features = model._run_forward(adv_data, mode='feature')
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.predict(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
                num_classes = model.roi_head.bbox_head.num_classes
            else:
                results = model.bbox_head.predict_by_feat(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
                num_classes = model.bbox_head.num_classes
            if results[0].bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results[0], init_results[0])
            loss = target_feature_loss(select_results, init_results[0], cls_logits, features, target_feature, num_classes)
            if loss == 0.0:
                break
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta 
        # return adv_images, grad_pre            
            # adv_images, grad_pre = self.attack_step(adv_images, model, target_loss, BI, data, init_results, ori_images, grad_pre)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    def select_targets(self, results, init_results):
        ious = bbox_overlaps(results.bboxes, init_results.bboxes)
        index = ious.max(1)[0] > 0.5
        return results[index]
        
    def get_target_feats(self, model, dataset, target_instance):
        per_class_target_feature = {}
        for ins in target_instance:
            label = ins.labels.item()
            if label in per_class_target_feature:
                continue
            else:
                support_img = random.choice(dataset.per_class_imgs[label + 1])
                anns = dataset.per_img_anns[support_img]
                target_anns = [ann for ann in anns if ann['category_id']== (label + 1)]
                target_ann = random.choice(target_anns)['bbox']
                target_ann = ann_xywh_to_xyxy(target_ann, 4)
                support_img = dataset.data_prefix.img + support_img
                sup_img = cv2.imread(support_img)
                sup_data = {
                    "inputs" : [torch.tensor(sup_img).permute(2,0,1).cuda()],
                }
                sup_data = model.data_preprocessor(sup_data, False)
                features = model._run_forward(sup_data, mode='feature')
                target_bbox_feature = features[..., target_ann[1]:target_ann[3], 
                                                target_ann[0]:target_ann[2]]
                # target_bbox_feature = F.interpolate(target_bbox_feature, bbox_feature.shape[2:], mode = 'bilinear', align_corners=False)
                per_class_target_feature[label] = target_bbox_feature
        return per_class_target_feature
        


@ATTACKERS.register_module()
class TABIMAttacker(TBIMAttacker):
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dota/images/P1128__1__0___480.png':
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/16914.png':
        #     print()
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            # features = model._run_forward(adv_data, mode='feature')
            features = None
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results, tar_instances, cls_logits, num_classes)
            loss = target_loss_v3(select_results, tar_instances)
            if loss == 0.0:
                print("attack failed")
                break
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
        # a = (adv_images - ori_images).abs().detach().cpu().numpy().sum(0)
        # bs = tar_instances.bboxes.cpu().numpy().astype(np.int64)
        # for b in bs:
        #     cv2.rectangle(a, (b[0], b[1]), (b[2],b[3]), color=(255))
        # cv2.imwrite("work_dirs/debug.png", a)
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data

    
    def select_targets(self, results, tar_instance, cls_logits, num_classes):
        select_results = []
        ious = bbox_overlaps(tar_instance.bboxes, results.bboxes)
        max_overlaps, argmax_overlaps = ious.max(0)
        for idx, tar in enumerate(tar_instance):
            ins = InstanceData()
            matched_overlaps = max_overlaps[argmax_overlaps == idx]
            matched_argmax_overlaps = argmax_overlaps[argmax_overlaps == idx]
            ins = results[argmax_overlaps == idx]
            logits = cls_logits[argmax_overlaps == idx]
            
            # matched_argmax_overlaps = matched_argmax_overlaps[matched_overlaps > 0.5]
            # ins = ins[matched_overlaps > 0.5]
            # logits = logits[matched_overlaps > 0.5]
            # matched_overlaps = matched_overlaps[matched_overlaps > 0.5]
            
            # matched_overlaps = matched_overlaps[ins.scores > 0.1]
            # logits = logits[ins.scores > 0.1]
            # ins = ins[ins.scores > 0.1]
            attributes = matched_overlaps * ins.scores
            cls_scores = F.cross_entropy(logits[:, :num_classes], \
                tar.labels.repeat(logits.shape[0]), reduction = 'none')
            weights = (1 - attributes) * cls_scores
            select_num = (weights < weights.mean()).sum()
            # select_num = weights.sum().long()
            _, sorted_indexes = torch.sort(weights)
            ins.weights = weights
            ins.iou = matched_overlaps
            ins.logits = logits
            ins.attributes = attributes
            ins.cost = cls_scores
            select_results.append(ins[sorted_indexes[:select_num]])
        return select_results
    

@ATTACKERS.register_module()
class TOGAttacker(TBIMAttacker):
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            # features = model._run_forward(adv_data, mode='feature')
            features = None
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results, tar_instances, cls_logits, num_classes)
            # loss = target_loss_v1(results[0], init_results[0], cls_logits, features, target_feature, num_classes)
            loss = target_loss_v2(select_results, tar_instances, features, num_classes)
            if loss == 0.0:
                print("attack failed")
                break
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    def select_targets(self, results, tar_instance, cls_logits, num_classes):
        select_results = []
        ious = bbox_overlaps(tar_instance.bboxes, results.bboxes)
        max_overlaps, argmax_overlaps = ious.max(0)
        for idx, tar in enumerate(tar_instance):
            ins = InstanceData()
            matched_overlaps = max_overlaps[argmax_overlaps == idx]
            matched_argmax_overlaps = argmax_overlaps[argmax_overlaps == idx]
            ins = results[argmax_overlaps == idx]
            logits = cls_logits[argmax_overlaps == idx]
            
            matched_argmax_overlaps = matched_argmax_overlaps[matched_overlaps > 0]
            ins = ins[matched_overlaps > 0]
            logits = logits[matched_overlaps > 0]
            matched_overlaps = matched_overlaps[matched_overlaps > 0]
            ins.iou = matched_overlaps
            ins.logits = logits
            select_results.append(ins)
        return select_results 


@ATTACKERS.register_module()
class TAIOUBIMAttacker(TBIMAttacker):
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dota/images/P1128__1__0___480.png':
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/16914.png':
        #     print()
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            # features = model._run_forward(adv_data, mode='feature')
            features = None
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results, tar_instances, cls_logits, num_classes)
            loss = target_loss_v4(select_results, tar_instances, features, num_classes)
            # loss = target_loss_withoutiou(select_results, tar_instances, features, num_classes)
            if loss == 0.0:
                print("attack failed")
                break            
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
        # a = (adv_images - ori_images).abs().detach().cpu().numpy().sum(0)
        # bs = tar_instances.bboxes.cpu().numpy().astype(np.int64)
        # for b in bs:
        #     cv2.rectangle(a, (b[0], b[1]), (b[2],b[3]), color=(255))
        # cv2.imwrite("work_dirs/debug.png", a)
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    
    def select_targets(self, results, tar_instance, cls_logits, num_classes):
        select_results = []
        ious = bbox_overlaps(tar_instance.bboxes, results.bboxes)
        max_overlaps, argmax_overlaps = ious.max(0)
        for idx, tar in enumerate(tar_instance):
            ins = InstanceData()
            matched_overlaps = max_overlaps[argmax_overlaps == idx]
            matched_argmax_overlaps = argmax_overlaps[argmax_overlaps == idx]
            ins = results[argmax_overlaps == idx]
            logits = cls_logits[argmax_overlaps == idx]
            
            matched_argmax_overlaps = matched_argmax_overlaps[matched_overlaps > 0.5]
            ins = ins[matched_overlaps > 0.5]
            logits = logits[matched_overlaps > 0.5]
            matched_overlaps = matched_overlaps[matched_overlaps > 0.5]
            
            matched_overlaps = matched_overlaps[ins.scores > 0.1]
            logits = logits[ins.scores > 0.1]
            ins = ins[ins.scores > 0.1]
            attributes = matched_overlaps * ins.scores
            cls_scores = F.cross_entropy(logits[:, :num_classes], \
                tar.labels.repeat(logits.shape[0]), reduction = 'none')
            weights = (1 - attributes) * cls_scores
            select_num = (weights < weights.mean()).sum()
            # select_num = weights.sum().long()
            _, sorted_indexes = torch.sort(weights)
            ins.weights = weights
            ins.iou = matched_overlaps
            ins.logits = logits
            select_results.append(ins[sorted_indexes[:select_num]])
        return select_results
    


@ATTACKERS.register_module()
class TCBIMAttacker(TBIMAttacker):
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dota/images/P1128__1__0___480.png':
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/16914.png':
        #     print()
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            # features = model._run_forward(adv_data, mode='feature')
            features = None
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results, tar_instances, cls_logits, num_classes)
            # loss = target_loss_v1(results[0], init_results[0], cls_logits, features, target_feature, num_classes)
            loss = target_loss_v4(select_results, tar_instances, features, num_classes)
            if loss == 0.0:
                print("attack failed")
                break
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
        # a = (adv_images - ori_images).abs().detach().cpu().numpy().sum(0)
        # bs = tar_instances.bboxes.cpu().numpy().astype(np.int64)
        # for b in bs:
        #     cv2.rectangle(a, (b[0], b[1]), (b[2],b[3]), color=(255))
        # cv2.imwrite("work_dirs/debug.png", a)
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data

    
    def select_targets(self, results, tar_instance, cls_logits, num_classes):
        select_results = []
        ious = bbox_overlaps(tar_instance.bboxes, results.bboxes)
        max_overlaps, argmax_overlaps = ious.max(0)
        for idx, tar in enumerate(tar_instance):
            ins = InstanceData()
            matched_overlaps = max_overlaps[argmax_overlaps == idx]
            matched_argmax_overlaps = argmax_overlaps[argmax_overlaps == idx]
            ins = results[argmax_overlaps == idx]
            logits = cls_logits[argmax_overlaps == idx]
            
            matched_argmax_overlaps = matched_argmax_overlaps[matched_overlaps > 0.5]
            ins = ins[matched_overlaps > 0.5]
            logits = logits[matched_overlaps > 0.5]
            matched_overlaps = matched_overlaps[matched_overlaps > 0.5]
            
            matched_overlaps = matched_overlaps[ins.scores > 0.1]
            logits = logits[ins.scores > 0.1]
            ins = ins[ins.scores > 0.1]
            attributes = matched_overlaps * ins.scores
            cls_scores = F.cross_entropy(logits[:, :num_classes], \
                tar.labels.repeat(logits.shape[0]), reduction = 'none')
            weights = (1 - attributes) * cls_scores
            select_num = (weights < weights.mean()).sum()
            # select_num = weights.sum().long()
            _, sorted_indexes = torch.sort(weights)
            ins.weights = weights
            ins.iou = matched_overlaps
            ins.logits = logits
            ins.attributes = attributes
            ins.cost = cls_scores
            select_results.append(ins[sorted_indexes[:select_num]])
        return select_results


@ATTACKERS.register_module()
class TALBIMAttacker(TBIMAttacker):
    
    def attack(self, model, data, dataset = None):
        # if data['data_samples'][0].img_path == '/disk2/lhd/datasets/attack/dior/images/16901.png':
        #     print()
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)
        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank = 5)
        
        if tar_instances == None:
            self.is_attack = False
            if not hasattr(self, 'count'):
                setattr(self, 'count', 0)
            self.count += 1
            print(f"{data['data_samples'][0].img_path[-9:]} has no valid detections!")
            return data
        else:
            self.is_attack = True
        attack_map = self.get_attack_map(tar_instances, ori_images)
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
            delta.requires_grad = True
            adv_data = {
                "inputs":[adv_images + delta],
                "data_samples":data['data_samples']
            }
            adv_data = model.data_preprocessor(adv_data, False)
            outs = model._run_forward(adv_data, mode='tensor')
            # features = model._run_forward(adv_data, mode='feature')
            features = None
            if hasattr(model, 'roi_head'):
                outs = [[out] for out in outs]
                results, cls_logits = model.roi_head.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], 
                            rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                results, cls_logits = model.bbox_head.inference(*outs, 
                            batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            if results.bboxes.shape[0] == 0:
                break
            select_results = self.select_targets(results, tar_instances, cls_logits, num_classes)
            # loss = target_loss_v1(results[0], init_results[0], cls_logits, features, target_feature, num_classes)
            loss = target_loss_v2(select_results, tar_instances, features, num_classes)
            loss.backward()
            noise = delta.grad.data.detach_().clone()
            noise, grad_pre = BI(noise, grad_pre)
            delta.grad.zero_()
            delta.data = delta.data - self.step_size * torch.sign(noise) * attack_map
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            
            adv_images = ori_images.clone() + delta
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    def select_targets(self, results, init_results, cls_logits, num_classes):
        select_results = []
        for init in init_results:
            ins = InstanceData()
            iou = bbox_overlaps(init.bboxes, results.bboxes)[0]
            index = iou > 0.5
            ins.bboxes = results.bboxes[index]
            ins.labels = results.labels[index]
            ins.scores = results.scores[index]
            ins.logits = cls_logits[index]
            ins.iou = iou[index]
            cls_scores = F.cross_entropy(cls_logits[index][:, :num_classes], \
                init.labels.repeat(cls_logits[index].shape[0]), reduction = 'none')
            sorted_scores, sorted_indexes = torch.sort(iou[index] * ins.scores * cls_scores, descending=True)
            # F.cross_entropy(cls_scores, init.labels.repeat(cls_scores.shape[0]), reduction = 'none')
            select_num = (sorted_scores > sorted_scores.mean()).sum()
            select_results.append(ins[sorted_indexes[:select_num]])
        # ious = bbox_overlaps(results.bboxes, init_results.bboxes)
        # index = ious.max(1)[0] > 0.3
        # return results[index]
        return select_results
    
    def get_attack_map(self, instances, images):
        attack_map = images.new_zeros(images.shape)
        for ins in instances:
            x1, y1, x2, y2 = list(map(int, ins.bboxes[0]))
            attack_map[:, y1:y2, x1:x2] = 1
        return attack_map
            


@ATTACKERS.register_module()
class TPBIMAttacker(TBIMAttacker):

    def attack_step(self, adv_images, model, loss_func, grad_func, attack_map, data, init_results, ori_images, grad_pre):
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        adv_data = {
            "inputs":[adv_images + delta],
            "data_samples":data['data_samples']
        }
        adv_data = model.data_preprocessor(adv_data, False)
        outs = model._run_forward(adv_data, mode='tensor')
        
        if hasattr(model, 'roi_head'):
            outs = [[out] for out in outs]
            results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            num_classes = model.roi_head.bbox_head.num_classes
        else:
            results = model.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            num_classes = model.bbox_head.num_classes
        if results[0].bboxes.shape[0] == 0:
            return adv_images, grad_pre
        loss = loss_func(results[0], init_results[0], cls_logits, num_classes)
        if loss == 0.0:
            return adv_images, grad_pre
        loss.backward()
        noise = delta.grad.data.detach_().clone()
        noise, grad_pre = grad_func(noise, grad_pre)
        if len(noise.shape) == 4:
            noise = noise[0]
        delta.grad.zero_()
        delta.data = delta.data - self.step_size * torch.sign(noise) * attack_map
        delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
        delta.data = delta.data.clamp(-self.eps, self.eps)
        
        adv_images = ori_images.clone() + delta 
        return adv_images, grad_pre
    
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)
        if init_results[0].bboxes.shape[0] == 0:
            return data
        attack_map = self.make_init_mask_img(ori_images, init_results[0])
        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
        if tar_instances == None:
            return data
        data['data_samples'][0].gt_instances = tar_instances
        init_results[0].labels.data = tar_instances.labels

        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            adv_images, grad_pre = self.attack_step(adv_images, model, target_bim_loss, BI, attack_map, data, init_results, ori_images, grad_pre)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    def make_init_mask_img(self, images, instance):
        attack_map = images.new_zeros(images.shape)
        for bbox in instance.bboxes.int():
            attack_map[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        return attack_map


@ATTACKERS.register_module()
class TMIMAttacker(TBIMAttacker):
 
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)
        if init_results[0].bboxes.shape[0] == 0:
            return data
        if self.targeted:
            
            tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
            if tar_instances == None:
                return data
            data['data_samples'][0].gt_instances = tar_instances
            init_results[0].labels.data = tar_instances.labels            
        else:
            raise NotImplementedError

        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            adv_images, grad_pre = self.attack_step(adv_images, model, target_class_loss_v1, MI, data, init_results, ori_images, grad_pre)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data


@ATTACKERS.register_module()
class TTIMAttacker(TBIMAttacker):
 
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)
        if init_results[0].bboxes.shape[0] == 0:
            return data
        if self.targeted:
            tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
            if tar_instances == None:
                return data
            data['data_samples'][0].gt_instances = tar_instances
            init_results[0].labels.data = tar_instances.labels            
        else:
            raise NotImplementedError

        adv_images = ori_images.clone()
        grad_pre = 0
        for ii in range(self.steps):
            adv_images, grad_pre = self.attack_step(adv_images, model, target_class_loss_v1, TI, data, init_results, ori_images, grad_pre)
     
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    
@ATTACKERS.register_module()
class TNIMAttacker(TBIMAttacker):
    
    def attack_step(self, adv_images, model, loss_func, grad_func, data, init_results, ori_images, grad_pre):
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        adv_data = {
            "inputs":[adv_images + delta - self.step_size * torch.sign(grad_pre)],
            "data_samples":data['data_samples']
        }
        adv_data = model.data_preprocessor(adv_data, False)
        outs = model._run_forward(adv_data, mode='tensor')
        
        if hasattr(model, 'roi_head'):
            outs = [[out] for out in outs]
            results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            num_classes = model.roi_head.bbox_head.num_classes
        else:
            results = model.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            num_classes = model.bbox_head.num_classes
        if results[0].bboxes.shape[0] == 0:
            return adv_images, grad_pre            
        loss = loss_func(results[0], init_results[0], cls_logits, num_classes)
        if loss == 0.0:
            return adv_images, grad_pre
        loss.backward()
        noise = delta.grad.data.detach_().clone()
        noise, grad_pre = grad_func(noise, grad_pre)
        delta.grad.zero_()
        delta.data = delta.data - self.step_size * torch.sign(noise)
        delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
        delta.data = delta.data.clamp(-self.eps, self.eps)
        adv_images = ori_images.clone() + delta 
        return adv_images, grad_pre
     
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)
        if init_results[0].bboxes.shape[0] == 0:
            return data
        if self.targeted:
            
            tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
            if tar_instances == None:
                return data
            data['data_samples'][0].gt_instances = tar_instances
            init_results[0].labels.data = tar_instances.labels            
        else:
            raise NotImplementedError

        adv_images = ori_images.clone()
        grad_pre = torch.tensor([0],device=torch.device('cuda'))
        for ii in range(self.steps):
            adv_images, grad_pre = self.attack_step(adv_images, model, target_class_loss_v1, MI, data, init_results, ori_images, grad_pre)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data
    
    
@ATTACKERS.register_module()
class TSIMAttacker(TBIMAttacker):
    
    def attack_step(self, adv_images, model, loss_func, grad_func, data, init_results, ori_images, grad_pre):
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        tmp_results = []
        for i in range(5):
            # scale = 1 if i==0 else pow(2, i)
            scale = pow(2, i)
            # if i == 0:
            #     scale = 1
            # else:
            #     scale = 2*i
            tmp_data = {
                "inputs":[(adv_images + delta) / scale],
                "data_samples":data['data_samples']
            }
            tmp_data = model.data_preprocessor(tmp_data, False)
            tmp_outs = model._run_forward(tmp_data, mode='tensor')
            tmp_outs = [[out] for out in tmp_outs]
            result, cls_logit = model.roi_head.bbox_head.predict_by_feat(*tmp_outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            tmp_results.append(result[0])
            
        adv_data = {
            "inputs":[adv_images + delta],
            "data_samples":data['data_samples']
        }
        adv_data = model.data_preprocessor(adv_data, False)
        outs = model._run_forward(adv_data, mode='tensor')
        
        if hasattr(model, 'roi_head'):
            outs = [[out] for out in outs]
            results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], 
                        rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            num_classes = model.roi_head.bbox_head.num_classes
        else:
            results = model.bbox_head.predict_by_feat(*outs, 
                        batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
            num_classes = model.bbox_head.num_classes
        loss = loss_func(results[0], init_results[0], cls_logits, num_classes)
        if loss == 0.0:
            return adv_images, grad_pre
        loss.backward()
        noise = delta.grad.data.detach_().clone()
        noise, grad_pre = grad_func(noise, grad_pre)
        delta.grad.zero_()
        delta.data = delta.data - self.step_size * torch.sign(noise)
        delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
        delta.data = delta.data.clamp(-self.eps, self.eps)
        adv_images = ori_images.clone() + delta 
        return adv_images, grad_pre
     
    def attack(self, model, data, dataset = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits = self.get_init_results(data, model)
        if init_results[0].bboxes.shape[0] == 0:
            return data
        # if self.targeted:
            
        #     tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results[0], cls_logits, rank = 5)
        #     if tar_instances == None:
        #         return data
        #     data['data_samples'][0].gt_instances = tar_instances
        #     init_results[0].labels.data = tar_instances.labels            
        # else:
        #     raise NotImplementedError

        adv_images = ori_images.clone()
        grad_pre = torch.tensor([0],device=torch.device('cuda'))
        for ii in range(self.steps):
            adv_images, grad_pre = self.attack_step(adv_images, model, target_class_loss, MI, data, init_results, ori_images, grad_pre)
        
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        return adv_data    