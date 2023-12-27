from abc import abstractmethod
import numpy as np
import copy
import torch
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

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        return repr_str    
    
# @ATTACKERS.register_module()
# class BIMAttacker(IterBaseAttacker):
    
#     def attack(self, model, data, dataset = None):
#         ori_images = data['inputs'][0].cuda()
#         model = self.prepare_model(model)
#         delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
#         delta.requires_grad = True
#         init_results, _ = self.get_init_results(data, model)
#         adv_images = ori_images.clone()
#         for ii in range(self.steps):
#             adv_images = self.attack_step(adv_images, model, class_loss, data, init_results, ori_images)
#         adv_data = {
#             "inputs":[adv_images],
#             "data_samples":data['data_samples']
#         } 
#         return adv_data

@ATTACKERS.register_module()
class TSIAAAttacker(IterBaseAttacker):
   
    def attack(self, model, data, dataset = None):
        np.random.seed(0)
        ori_images = data['inputs'][0].cuda()
        model = self.prepare_model(model)
        delta = torch.zeros(data['inputs'][0].shape, dtype = torch.float32, device = torch.device('cuda'))
        delta.requires_grad = True
        init_results, cls_logits, num_classes = self.get_init_results(data, model)

        tar_instances = get_target_instance(data['data_samples'][0].gt_instances, init_results, cls_logits, num_classes, rank_type = 'random')
        if tar_instances == None:
            self.is_attack = False
            return data
        else:
            self.is_attack = True
        data['data_samples'][0].gt_instances = tar_instances
        init_results.labels.data = tar_instances.labels
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
                break
            loss.backward()
            noise = delta.grad.data.detach_().clone()
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
