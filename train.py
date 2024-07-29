import os
import argparse
import pickle
from collections import defaultdict
import sklearn.metrics
from scipy.stats import gaussian_kde

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from configs import cfg, update_config

from dataset.augmentation import get_transform
from dataset.par import PEDES_DATASET, ParDataset

from batch_engine import valid_trainer, batch_trainer, test_trainer

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

seed = int(np.random.choice(1000, 1))
# seed = 605
set_seed(seed)


class ParTrainer:
    def __init__(self, attr_filter = []):
        '''loading dataset and model'''
        
        # defines path
        self.timestamp = time_str()
        self.exp_dpath = os.path.join('exp_result', cfg.DATASET.NAME)
        self.model_fpath, self.log_fpath = get_model_log_path(self.exp_dpath, cfg.BACKBONE.TYPE+cfg.NAME)
        self.save_model_fpath = os.path.join(self.model_fpath, f'ckpt_max_{self.timestamp}.pth')
        gen_code_archive(out_dir=os.path.join(self.exp_dpath, cfg.BACKBONE.TYPE+cfg.NAME), file=f'code_{self.timestamp}.tar.gz')

        # define data transform
        train_tsfm, valid_tsfm = get_transform(cfg)


        self.train_set: ParDataset = PEDES_DATASET[cfg.DATASET.NAME](
            cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
            target_transform=cfg.DATASET.TARGETTRANSFORM, attrs=attr_filter)

        self.valid_set: ParDataset = PEDES_DATASET[cfg.DATASET.NAME](
            cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
            target_transform=cfg.DATASET.TARGETTRANSFORM, attrs=attr_filter)
        
        self.test_set: ParDataset = PEDES_DATASET['adddigencls'](
            data_dpath='/home/adddai/ex_hdd/datasets/adddi/par/testset_240723', transform=valid_tsfm,
            target_transform=cfg.DATASET.TARGETTRANSFORM)

        
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
        
        self.test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        print('-' * 60)
        print(f'{cfg.DATASET.NAME} attr_num : {len(self.train_set.attrs)}\n'
              f'{cfg.DATASET.TRAIN_SPLIT} set: {len(self.train_loader.dataset)}, \n'
              f'{cfg.DATASET.TEST_SPLIT} set: {len(self.valid_loader.dataset)}, \n'
              f'gender classification(custom) test set: {len(self.test_loader.dataset)}, \n'
        )
        
        backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


        classifier = build_classifier(cfg.CLASSIFIER.NAME)(
            nattr=len(self.train_set.attrs),
            c_in=c_output,
            bn=cfg.CLASSIFIER.BN,
            pool=cfg.CLASSIFIER.POOLING,
            scale =cfg.CLASSIFIER.SCALE
        )

        self.model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)

        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        
        self.model_ema = None
        if cfg.TRAIN.EMA.ENABLE:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEmaV2(
                self.model, decay=cfg.TRAIN.EMA.DECAY, device='cpu' if cfg.TRAIN.EMA.FORCE_CPU else None)

    def __get_train_properties(self):
        if cfg.RELOAD.TYPE:
            self.model = get_reload_weight(self.model_fpath, self.model, pth=cfg.RELOAD.PTH)

        labels = self.train_set.get_labels()
        label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None
        print('label_ratio:', label_ratio)
        criterion = build_loss(cfg.LOSS.TYPE)(
            sample_weight=label_ratio, scale=cfg.CLASSIFIER.SCALE, size_sum=cfg.LOSS.SIZESUM)
        criterion = criterion.cuda()
        
        if cfg.TRAIN.BN_WD:
            param_groups = [{'params': self.model.module.finetune_params(),
                            'lr': cfg.TRAIN.LR_SCHEDULER.LR_FT,
                            'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY},
                            {'params': self.model.module.fresh_params(),
                            'lr': cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                            'weight_decay': cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY}]
        else:
            # bn parameters are not applied with weight decay
            ft_params = seperate_weight_decay(
                self.model.module.finetune_params(),
                lr=cfg.TRAIN.LR_SCHEDULER.LR_FT,
                weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

            fresh_params = seperate_weight_decay(
                self.model.module.fresh_params(),
                lr=cfg.TRAIN.LR_SCHEDULER.LR_NEW,
                weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY)

            param_groups = ft_params + fresh_params

        if cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'sgd':
            optimizer = torch.optim.SGD(param_groups, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM)
        elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adam':
            optimizer = torch.optim.Adam(param_groups)
        elif cfg.TRAIN.OPTIMIZER.TYPE.lower() == 'adamw':
            optimizer = AdamW(param_groups)
        else:
            assert None, f'{cfg.TRAIN.OPTIMIZER.TYPE} is not implemented'

        if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
            if cfg.CLASSIFIER.BN:
                assert False, 'BN can not compatible with ReduceLROnPlateau'
        elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
            lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_SCHEDULER.LR_STEP, gamma=0.1)
        elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine':

            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=cfg.TRAIN.MAX_EPOCH * len(self.train_loader),
                lr_min=1e-6,  # cosine lr 最终回落的位置
                warmup_t=cfg.TRAIN.MAX_EPOCH * len(self.train_loader) * cfg.TRAIN.LR_SCHEDULER.WMUP_COEF,
                warmup_lr_init=cfg.TRAIN.LR_SCHEDULER.WMUP_LR_INIT,
                t_mul=1.,
                decay_rate=0.1,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=None,
                noise_pct=0.67,
                noise_std=1.,
                noise_seed=42,
            )

        else:
            assert False, f'{cfg.TRAIN.LR_SCHEDULER.TYPE} has not been achieved yet'
        return criterion, optimizer, lr_scheduler

    def train(self):
        maximum_auroc = float(-np.inf)
        minimum_iou = float(np.inf)
        maximum_compmetric = float(-np.inf)
        
        best_epoch = 0

        result_list = defaultdict()

        result_path = self.save_model_fpath
        result_path = result_path.replace('ckpt_max', 'metric')
        result_path = result_path.replace('pth', 'pkl')

        criterion, optimizer, lr_scheduler = self.__get_train_properties()
        
        for e in range(cfg.TRAIN.MAX_EPOCH):
            lr = optimizer.param_groups[1]['lr']

            train_loss, train_gt, train_probs, train_imgs, train_logits, train_loss_mtr = batch_trainer(
                cfg,
                args=args,
                epoch=e,
                model=self.model,
                model_ema=self.model_ema,
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                loss_w=cfg.LOSS.LOSS_WEIGHT,
                scheduler=lr_scheduler if cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmup_cosine' else None,
            )


            if self.model_ema is not None and not cfg.TRAIN.EMA.FORCE_CPU:
                valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
                    cfg,
                    args=args,
                    epoch=e,
                    model=self.model_ema.module,
                    valid_loader=self.valid_loader,
                    criterion=criterion,
                    loss_w=cfg.LOSS.LOSS_WEIGHT
                )
            else:
                valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
                    cfg,
                    args=args,
                    epoch=e,
                    model=self.model,
                    valid_loader=self.valid_loader,
                    criterion=criterion,
                    loss_w=cfg.LOSS.LOSS_WEIGHT
                )
                test_gt, test_probs, test_imgs, _ = test_trainer(
                    cfg,
                    args=args,
                    epoch=e,
                    model=self.model,
                    valid_loader=self.test_loader,
                    criterion=criterion,
                    loss_w=cfg.LOSS.LOSS_WEIGHT
                )

            if cfg.TRAIN.LR_SCHEDULER.TYPE == 'plateau':
                lr_scheduler.step(metrics=valid_loss)
            elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'multistep':
                lr_scheduler.step()

            if cfg.METRIC.TYPE == 'pedestrian':
                train_result = get_pedestrian_metrics(train_gt, train_probs, index=None)
                valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None)

                print(f'Evaluation on train set, train losses {train_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        train_result.ma, np.mean(train_result.label_f1),
                        np.mean(train_result.label_pos_recall),
                        np.mean(train_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                        train_result.instance_f1))

                print(f'Evaluation on test set, valid losses {valid_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        valid_result.ma, np.mean(valid_result.label_f1),
                        np.mean(valid_result.label_pos_recall),
                        np.mean(valid_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                        valid_result.instance_f1))

                print(f'{time_str()}')
                print('-' * 60)
                
                '''auroc per class'''
                print(f'AUC-ROC on dataset: ')
                for attr in self.train_set.attrs:
                    attr_idx = self.train_set.idxof_attr(attr)
                    train_auroc = sklearn.metrics.roc_auc_score(y_true=train_gt[:, attr_idx], y_score=train_probs[:, attr_idx])
                    valid_auroc = sklearn.metrics.roc_auc_score(y_true=valid_gt[:, attr_idx], y_score=valid_probs[:, attr_idx])
                    
                    if attr == 'gender_female':
                        lbd_test_gt = test_gt[test_gt < 1.1]
                        lbd_test_probs = test_probs[test_gt[:, 0] < 1.1, :]
                        y_tarr = lbd_test_gt
                        y_sarr = lbd_test_probs[:, attr_idx]
                        test_auroc = sklearn.metrics.roc_auc_score(y_true=y_tarr, y_score=y_sarr)
                        
                        female_inds = y_tarr > .5
                        female_data = y_sarr[female_inds]
                        male_data = y_sarr[~female_inds]
                        
                        kde_female = gaussian_kde(female_data)
                        kde_male = gaussian_kde(male_data)
                        
                        n_samples = 1000
                        x = np.linspace(min(female_data.min(), male_data.min()), max(female_data.max(), male_data.max()), n_samples)
                        
                        # 커널 밀도 추정 값을 계산
                        kde_female_values = kde_female(x)
                        kde_male_values = kde_male(x)
                        intersection = np.minimum(kde_female_values, kde_male_values)
                        intersection_area = np.trapz(intersection, x)
                        max_intersection_index = np.argmax(intersection) / n_samples
                        union = np.maximum(kde_female_values, kde_male_values)
                        union_area = np.trapz(union, x)
                        iou_crit = intersection_area / union_area
                        
                        print(f'[{attr}] train: {train_auroc}, valid: {valid_auroc}, test: {test_auroc}, test_iou: {iou_crit}, opt-thresh: {max_intersection_index}')
                    else:
                        print(f'[{attr}] train: {train_auroc}, valid: {valid_auroc}')
                        
                print('-' * 60)
                    
                if test_auroc > maximum_auroc:
                    maximum_auroc = test_auroc
                    best_epoch = e
                    save_fpath = os.path.join(self.model_fpath, f'ckpt_max_{self.timestamp}_auroc{test_auroc:.3f}.pth')
                    print(f'※ 최고 AUROC 갱신, 가중치를 저장합니다: {save_fpath}')
                    save_ckpt(self.model, save_fpath, e, maximum_auroc)
                
                if (compmetric := (test_auroc - iou_crit)) > maximum_compmetric:
                    maximum_compmetric = compmetric
                    best_epoch = e
                    save_fpath = os.path.join(self.model_fpath, f'ckpt_max_{self.timestamp}_compmetric{compmetric:.3f}.pth')
                    print(f'※ 최고 AUROC - IOU 갱신, 가중치를 저장합니다: {save_fpath}')
                    save_ckpt(self.model, save_fpath, e, maximum_compmetric)

                if iou_crit < minimum_iou:
                    minimum_iou = iou_crit
                    best_epoch = e
                    save_fpath = os.path.join(self.model_fpath, f'ckpt_max_{self.timestamp}_iou{iou_crit:.3f}.pth')
                    print(f'※ 최저 예측분포 IOU 갱신, 가중치를 저장합니다: {save_fpath}')
                    save_ckpt(self.model, save_fpath, e, minimum_iou)
                
                print('-' * 60)
                
                result_list[e] = {
                    'train_result': train_result,  # 'train_map': train_map,
                    'valid_result': valid_result,  # 'valid_map': valid_map,
                    'train_gt': train_gt, 'train_probs': train_probs,
                    'valid_gt': valid_gt, 'valid_probs': valid_probs,
                    'train_imgs': train_imgs, 'valid_imgs': valid_imgs
                }
            else:
                assert False, f'{cfg.METRIC.TYPE} is unavailable'

            with open(result_path, 'wb') as f:
                pickle.dump(result_list, f)

        return maximum_auroc, best_epoch


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
    trainer = ParTrainer(['gender_female', 'pose_fore'])
    best_metric, epoch = trainer.train()
    print(f'{cfg.NAME},  best_metrc : {best_metric} in epoch{epoch}')