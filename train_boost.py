import os
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from configs import cfg, update_config

from dataset.augmentation import get_transform
from dataset.par import PEDES_DATASET, ParDataset

from batch_engine import valid_trainer, batch_trainer

from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone
from models.model_ema import ModelEmaV2
from ops.optim.adamw import AdamW
from ops.scheduler.cosine_lr import CosineLRScheduler
from metrics.pedestrian_metrics import get_pedestrian_metrics

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

# seed = int(np.random.choice(1000, 1))
seed = 605
set_seed(seed)


class ParBoostTrainer:
    def __init__(self, attr_filter = []):
        '''loading dataset and model'''
        
        # defines path
        timestamp = time_str()
        self.exp_dpath = os.path.join('exp_result', cfg.DATASET.NAME)
        self.model_fpath, self.log_fpath = get_model_log_path(self.exp_dpath, cfg.BACKBONE.TYPE+cfg.NAME)
        self.save_model_fpath = os.path.join(self.model_fpath, f'ckpt_max_{timestamp}.pth')
        gen_code_archive(out_dir=os.path.join(self.exp_dpath, cfg.BACKBONE.TYPE+cfg.NAME), file=f'code_{timestamp}.tar.gz')

        # define data transform
        train_tsfm, valid_tsfm = get_transform(cfg)


        self.train_set: ParDataset = PEDES_DATASET[cfg.DATASET.NAME](
            cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
            target_transform=cfg.DATASET.TARGETTRANSFORM, attrs=attr_filter)

        self.valid_set: ParDataset = PEDES_DATASET[cfg.DATASET.NAME](
            cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
            target_transform=cfg.DATASET.TARGETTRANSFORM, attrs=attr_filter)
        
        
        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=None,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            dataset=self.valid_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print('-' * 60)
        print(f'{cfg.DATASET.NAME} attr_num : {len(self.train_set.attrs)}\n'
              f'{cfg.DATASET.TRAIN_SPLIT} set: {len(self.train_loader.dataset)}, \n'
              f'{cfg.DATASET.TEST_SPLIT} set: {len(self.valid_loader.dataset)}, \n'
        )
        
        backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
        
        self.feature_extractor = backbone

        self.feature_extractor = self.feature_extractor.cuda()
        self.feature_extractor = torch.nn.DataParallel(self.feature_extractor)
        
    def train(self):
        for batch in self.train_loader:
            print(batch[1].shape)
            out = self.feature_extractor(batch[0])
            print(out.shape)
            exit()


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/adddi.yaml",
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args.cfg)
    trainer = ParBoostTrainer(['gender_female'])
    trainer.train()
    # print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')