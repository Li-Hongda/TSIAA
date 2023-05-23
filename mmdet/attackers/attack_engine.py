from mmengine.registry import RUNNERS, LOOPS
from mmengine.runner.loops import TestLoop
from mmengine.runner.runner import Runner

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn

from mmdet.registry import ATTACKERS

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
        self.attack_step(data_batch)
        # adv_batch = self.attack_step(data_batch)
        # outputs = self.runner.model.test_step(adv_batch)
        # self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=None)
    
    def attack_step(self, data_batch):
        gen_model = self.runner.model
        # setattr(gen_model.bbox_head, "attack", "True")
        adv_batch = self.runner.attacker.attack(gen_model, data_batch)
        # return adv_batch