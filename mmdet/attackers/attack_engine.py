import cv2
import json
import numpy as np
import os

import mmengine
from mmengine.registry import RUNNERS, LOOPS, HOOKS
from mmengine.runner.loops import TestLoop
from mmengine.runner.runner import Runner
from mmengine.hooks.hook import DATA_BATCH, Hook

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn

from mmdet.registry import ATTACKERS

@HOOKS.register_module()
class RecordHook(Hook):
    def __init__(self, output_dir, with_ann = True):
        self.output_dir = output_dir
        self.with_ann = with_ann
        
    def after_test_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Optional[Sequence] = None) -> None:
        data_sample = data_batch['data_samples'][0]
        images = data_batch['inputs'][0]        
        if batch_idx == 0:
            self.dataset_name = data_sample.img_path.strip().split('/')[5]
            self.model_name = runner.model.__class__.__name__.lower()
            self.attacker_name = runner.attacker.__class__.__name__[:-8].lower()
            self.image_path = f'work_dirs/examples/{self.dataset_name}_{self.attacker_name}_{self.model_name}/images'
            self.label_path = f'work_dirs/examples/{self.dataset_name}_{self.attacker_name}_{self.model_name}/labels'
            mmengine.mkdir_or_exist(self.image_path)
            mmengine.mkdir_or_exist(self.label_path)
        img_name =  data_sample.img_path.strip().split('/')[-1]
        adv_images = images.cpu().detach().numpy().astype(np.uint8).transpose(1,2,0)
        cv2.imwrite(f'{self.image_path}/{img_name}',adv_images)
        if self.with_ann:
            bboxes = data_sample.gt_instances.bboxes.tensor
            labels = data_sample.gt_instances.labels
            with open(f'{self.label_path}/{img_name[:-3]}txt', 'a') as f:
                for i in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = bboxes[i].tolist()
                    category = runner.test_dataloader.dataset.METAINFO['classes'][labels[i].item()]
                    outline = ' '.join(list(map(str, [x1, y1, x2, y1, x2, y2, x1, y2, category, 0, '\n'])))
                    f.write(outline)

    def after_test(self, runner) -> None:

        destfile = self.image_path[:-7] + '/select.json'
        # imageparent = os.path.join(self.image_path)
        labelparent = os.path.join(self.label_path)
        
        data_dict = {}
        data_dict['images'] = []
        data_dict['categories'] = []
        data_dict['annotations'] = []
        for idex, name in enumerate(runner.test_dataloader.dataset.METAINFO['classes']):
            single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
            data_dict['categories'].append(single_cat)

        inst_count = 1
        image_id = 1
        with open(destfile, 'w') as f_out:
            filenames = os.listdir(labelparent)
            for file in filenames:
                basename = os.path.basename(file)
                single_image = {}
                single_image['file_name'] = basename[:-4] + '.png'
                single_image['id'] = image_id
                single_image['width'] = 800
                single_image['height'] = 800
                data_dict['images'].append(single_image)

                # annotations
                with open(os.path.join(self.label_path, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        single_obj = {}
                        splitline = line.strip().split(' ')
                        poly = list(map(int, map(float, splitline[:8])))
                        cat = splitline[8]
                        single_obj['category_id'] = runner.test_dataloader.dataset.METAINFO['classes'].index(cat) + 1
                        single_obj['segmentation'] = []
                        single_obj['segmentation'].append(poly)
                        xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), \
                                                max(poly[0::2]), max(poly[1::2])
                        splitline = line.strip().split(' ')
                        width, height = xmax - xmin, ymax - ymin
                        single_obj['bbox'] = xmin, ymin, width, height
                        single_obj['area'] = float((width * height))
                        single_obj['image_id'] = image_id
                        data_dict['annotations'].append(single_obj)
                        single_obj['id'] = inst_count
                        inst_count = inst_count + 1
                image_id = image_id + 1 
            json.dump(data_dict, f_out)
    

@RUNNERS.register_module()
class AttackRunner(Runner):
    def __init__(self,
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        attacker_cfg = kwargs.get('cfg').get('attacker')
        self.attacker = self.build_attakcer(attacker_cfg)
    
    def build_attakcer(self, attacker: Union[nn.Module, Dict]) -> nn.Module:
        if isinstance(attacker, nn.Module):
            return attacker
        elif isinstance(attacker, dict):
            attacker = ATTACKERS.build(attacker)
            return attacker  # type: ignore
        else:
            raise TypeError('attacker should be a nn.Module object or dict, '
                            f'but got {attacker}') 
            
    def attack(self) -> dict:
        self.attack_loop = LOOPS.build(
                dict(type='AttackLoop'),
                default_args=dict(
                    runner=self,
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator))
        self.load_or_resume()
        self.call_hook('before_run')
        adv_batch = self.attack_loop.run()
        self.call_hook('after_run')
        return adv_batch
        
        
    


@LOOPS.register_module()
class AttackLoop(TestLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        # metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch')
        self.runner.call_hook('after_test')

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        # with autocast(enabled=self.fp16):
        gen_model = self.runner.model
        adv_batch = self.runner.attacker.attack(gen_model, data_batch)
        # outputs = self.runner.model.test_step(adv_batch)
        # self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=adv_batch,
            outputs=None)
        
        