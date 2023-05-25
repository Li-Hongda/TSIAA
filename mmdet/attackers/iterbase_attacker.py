from abc import abstractmethod
import numpy as np
import copy
import cv2
import torch
import mmcv
import mmengine
from mmdet.registry import ATTACKERS
import torch.nn as nn
from .utils import get_target_label, region_img_mean, k_largest_index_argsort, mask_img_result_change
from .loss import *


@ATTACKERS.register_module()
class IterBaseAttacker(nn.Module):    
    def __init__(self,
                 img_mean, 
                 img_std, 
                 steps = 20,
                 step_size = 1,
                 epsilon = 16,
                 targeted = False):
        
        self.eps = epsilon
        self.step_size = step_size
        self.steps = steps        
        self.mean = torch.tensor(img_mean, device = 'cuda')
        self.std = torch.tensor(img_std, device = 'cuda')
        self.targeted = targeted

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
    
    def visualize_samples(self, images, model, data_sample):
        dataset_name = data_sample.img_path.strip().split('/')[5]
        model_name = model.__class__.__name__.lower()
        attacker_name = self.__class__.__name__[:-8].lower()
        mmengine.mkdir_or_exist(f'work_dirs/examples/{dataset_name}_{attacker_name}_{model_name}_step{self.steps}') 
        img_name =  data_sample.img_path.strip().split('/')[-1]
        adv_images = images.cpu().detach().numpy().astype(np.uint8).transpose(1,2,0)
        cv2.imwrite(f'work_dirs/examples/{dataset_name}_{attacker_name}_{model_name}_step{self.steps}/{img_name}',adv_images)    

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


    def make_init_mask_img_torch(self, ori_images, init_results, model):
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
                            
                                boxes_mask , labels_mask = faster_rcnn_attack_box(img_path, faster_rcnn_model, img_mask, img_mask)
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
    
    def attack(self, model, data, anntations = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, _ = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                init_results = model.bbox_head.predict_by_feat(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
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
            delta.data = delta.data + self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            adv_images = ori_images.clone() + delta
            del loss, delta
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }        
        self.visualize_samples(adv_images, model, data['data_samples'][0])
        return adv_data
    

@ATTACKERS.register_module()
class BIMFODAttacker(IterBaseAttacker):
    
    def attack(self, model, data, anntations = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, _ = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                init_results = model.bbox_head.predict_by_feat(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], rescale=True)
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
        self.visualize_samples(adv_images, model, data['data_samples'][0])
        return adv_data


@ATTACKERS.register_module()
class BIMIOUAttacker(BIMAttacker):
    
    def attack(self, model, data, save = False):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_img['inputs'].requires_grad = True
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, _ = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                init_results = model.bbox_head.predict_by_feat(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], 
                               rescale=True)
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
            delta.data = delta.data + self.step_size * torch.sign(noise)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            delta.data = delta.data.clamp(-self.eps, self.eps)
            adv_images = ori_images.clone() + delta
            del loss, class_loss, iou_loss, delta
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }        
        self.visualize_samples(adv_images, model, data['data_samples'][0])
        return adv_data


@ATTACKERS.register_module()
class BIMIOUFODAttacker(BIMAttacker):
    
    def attack(self, model, data, save = False):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_img['inputs'].requires_grad = True
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, _ = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                  batch_img_metas=[data['data_samples'][0].metainfo], 
                                  rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                init_results = model.bbox_head.predict_by_feat(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], 
                               rescale=True)
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
        self.visualize_samples(adv_images, model, data['data_samples'][0])
        return adv_data


@ATTACKERS.register_module()
class TBIMAttacker(BIMAttacker):
    def __init__(self, targeted = True, **kwargs): 
        super().__init__(**kwargs)
        self.targeted = targeted
    
    def attack(self, model, data, anntations = None):
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        with torch.no_grad():
            init_img = model.data_preprocessor(copy.deepcopy(data), False)
            init_img['inputs'].requires_grad = True
            init_outs = model._run_forward(init_img, mode='tensor')
            if hasattr(model, 'roi_head'):
                init_outs = [[out] for out in init_outs]
                init_results, cls_logits = model.roi_head.bbox_head.predict_by_feat(*init_outs, 
                                           batch_img_metas=[data['data_samples'][0].metainfo], 
                                           rcnn_test_cfg = model.roi_head.test_cfg, rescale=True)
            else:
                init_results = model.bbox_head.predict_by_feat(*init_outs, 
                               batch_img_metas=[data['data_samples'][0].metainfo], 
                               rescale=True)
            if self.targeted:
                tar_labels = get_target_label(cls_logits, 5)
                tar_instances = data['data_samples'][0].gt_instances
                tar_instances.bboxes.tensor = init_results[0].bboxes
                tar_instances.labels.data = tar_labels
                data['data_samples'][0].gt_instances = tar_instances
            else:
                raise NotImplementedError

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
            class_loss.backward()
            noise = delta.grad.data.detach_().clone()
            delta.grad.zero_()
            delta.data = delta.data + self.step_size * torch.sign(noise)
            delta.data = delta.data.clamp(-self.eps, self.eps)
            delta.data = ((adv_images + delta.data).clamp(0, 255)) - ori_images
            
            adv_images = ori_images.clone() + delta
            del loss, class_loss, iou_loss, delta
            
        adv_data = {
            "inputs":[adv_images],
            "data_samples":data['data_samples']
        }
        # self.visualize_samples(adv_images, model, data['data_samples'][0])
        return adv_data
