import glob
import os
import pickle
import random
from abc import *

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from configs.default import BasicCN
from .dataset_registry import PEDES_DATASET

'''
█     █▄▄ ▀ ▄▄    █▄▀ ▀ ▄▄▄▄
█▀█ ▄ █▄█ █ █ █ ▄ █ █ █ █ █ █
'''
DATASPLIT_SEED = 42
random.seed(DATASPLIT_SEED)

class CheckStatus:
    __Passed = 1
    __Unknown = 0
    __NotPassed = -1

    def __init__(self):
        self.__status = self.Unknown
        
    @property
    def Passed(self):
        return self.__Passed
    @property
    def Unknown(self):
        return self.__Unknown
    @property
    def NotPassed(self):
        return self.__NotPassed
    
    @property
    def status(self):
        return self.__status
    
    def is_passed(self):
        return self.status == self.__Passed
    
    def set_status(self, flag):
        if flag:
            self.__status = self.Passed
        else:
            self.__status = self.NotPassed
    
    def get_check_icon(self):
        if self.status == self.Passed:
            return '🟢'
        if self.status == self.Unknown:
            return '🟠'
        if self.status == self.NotPassed:
            return '🔴'
    

class ParDataset(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def get_labels(self) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def attrs(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def idxof_attr(self, attr:str):
        pass
    

class AdddiParDataset(ParDataset):
    PARTITION_NAMES = ['train', 'val', 'test']
    
    def __init__(self, cfg: BasicCN, split:str, balance:bool=True, transform=None, target_transform=None, attrs=[]):
        super().__init__()
        self.cfg = cfg
        self.image_dpath = os.path.join(self.cfg.DATASET.PATH, 'data')
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.once_balanced = False
        self.label_key = f'{self.split}_label'
        self.data_key = f'{self.split}_images_name'
        assert self.check_dataset()
        
        from scipy import io
        mat_path = os.path.join(self.cfg.DATASET.PATH, 'annotation.mat')
        self.__mat = io.loadmat(mat_path)
        
        if len(attrs) < 1:
            self.__attrs = [i[0] for i in self.__mat['attributes'][:, 0]]
            self.attr_inds = list(range(len(self.__attrs)))
        else:
            exist_attrs = [i[0] for i in self.__mat['attributes'][:, 0]]
            assert all([a in exist_attrs for a in attrs]), 'There is an incorrect item in param::attr.'
            self.__attrs = attrs
            self.attr_inds = [self.idxof_attr(a) for a in self.__attrs]
        
        self.__label_arr = self.__mat[self.label_key][:, self.attr_inds]
        self.__imagefn_arr = np.array([fn[0] for fn in self.__mat[self.data_key][:, 0]])
                
        if balance:
            self.__balanced_dataset()
    
    def __balanced_dataset(self, criterion='gender_female'):
        if not self.once_balanced:
            self.once_balanced = True
            assert criterion in self.attrs
            crit_labels = [i[self.idxof_attr(criterion)] for i in self.__mat[self.label_key]]
            
            if more_true := (n_f := (len(crit_labels) - sum(crit_labels)) < (n_t := sum(crit_labels))):
                balance_k = n_f
            else:
                balance_k = n_t
            
            inds = list(range(self.__mat[self.label_key].shape[0]))
            more_inds = []
            less_inds = []
            for idx in inds:
                if crit_labels[idx] == int(more_true):
                    more_inds.append(idx)
                else:
                    less_inds.append(idx)
            balanced_mores = random.sample(more_inds, balance_k)
            comb_inds = less_inds + balanced_mores
            self.__label_arr = self.__label_arr[comb_inds]
            self.__imagefn_arr = self.__imagefn_arr[comb_inds]
    
    def check_dataset(self):
        print('━━━━━━──────┄┄┄┄┄┄┈┈┈┈┈┈')
        print('Checking dataset...')
        checklist = {
            'scipy_installed': CheckStatus(),
            'data_path_exist': CheckStatus(),
            'annotation_exist': CheckStatus(),
            'image_existence': CheckStatus(),
            'annotation_valid': CheckStatus(),
        }
        essential_datakeys = [
            'train_images_name', 'val_images_name', 'test_images_name', 
            'train_label', 'val_label', 'test_label', 
            'attributes']
        
        try:
            from scipy import io
            checklist['scipy_installed'].set_status(True)
        except Exception:
            pass
        
        checklist['data_path_exist'].set_status(os.path.isdir(self.cfg.DATASET.PATH))
        checklist['annotation_exist'].set_status(os.path.isfile(mat_path := os.path.join(self.cfg.DATASET.PATH, 'annotation.mat')))
        
        if checklist['scipy_installed'].is_passed() and checklist['annotation_exist'].is_passed():  #  annotation file open-able
            mat = io.loadmat(mat_path)
            checklist['annotation_valid'].set_status(all([k in mat.keys() for k in essential_datakeys]))
            if checklist['annotation_valid'].is_passed():
                image_names = np.concatenate([
                    [i[0] for i in mat['train_images_name'][:, 0]],
                    [i[0] for i in mat['val_images_name'][:, 0]],
                    [i[0] for i in mat['test_images_name'][:, 0]]], axis=0)
                checklist['image_existence'].set_status(all([os.path.isfile(os.path.join(self.image_dpath, fn)) for fn in image_names]))
                
        max_strlen = max([len(ckk) for ckk in checklist.keys()])
        lineheads = ['-'] * len(checklist.keys())
        for linehead, (ckk, ckv) in zip(lineheads, checklist.items()):
            padded = ckk.zfill(max_strlen).replace('0', ' ')
            print(f'{linehead} {padded}: {ckv.get_check_icon()}')
        print('━━━━━━──────┄┄┄┄┄┄┈┈┈┈┈┈')
        
        return all(checklist.values())
    
    def get_labels(self):
        return self.__label_arr.copy()
    
    @property
    def attrs(self):
        return self.__attrs
    
    def idxof_attr(self, attr:str):
        if attr in self.attrs:
            return self.attrs.index(attr)
        else:
            return -1
    
    def __getitem__(self, index):
        image_fname, gt_label = self.__imagefn_arr[index], self.__label_arr[index]
        image_fpath = os.path.join(self.image_dpath, image_fname)
        if os.path.isdir(image_fpath):
            frame_list = os.listdir(image_fpath)
            selected_frame_name = random.choice(frame_list)
            img = Image.open(os.path.join(image_fpath, selected_frame_name))
        elif os.path.isfile(image_fpath):
            img = Image.open(image_fpath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform:
            gt_label = gt_label[self.target_transform]
        return img, gt_label, image_fname
    
    def __len__(self):
        return self.__label_arr.shape[0]


class AdddiGenClsTestDataset(ParDataset):
    PARTITION_NAMES = ['test']
    CLASSES = ['male', 'female', 'person']
    IMG_EXTS = ['.png', '.jpg']
    
    def __init__(self, data_dpath:str, transform=None, target_transform=None):
        super().__init__()
        self.data_dpath = data_dpath
        
        self.male_dpath = os.path.join(self.data_dpath, self.CLASSES[0])
        self.female_dpath = os.path.join(self.data_dpath, self.CLASSES[1])
        self.uncls_dpath = os.path.join(self.data_dpath, self.CLASSES[2])
        
        self.transform = transform
        self.target_transform = target_transform
        assert self.check_dataset()
        
        image_fpaths = []
        label_list = []
        for cls_dname in self.CLASSES:
            cls_dpath = os.path.join(self.data_dpath, cls_dname)
            for img_fname in os.listdir(cls_dpath):
                image_fpaths.append(os.path.join(self.data_dpath, cls_dname, img_fname))
                label_list.append([self.CLASSES.index(cls_dname)])  # if person, return 2
        self.__label_arr = np.array(label_list, np.float32)
        self.__imagefp_arr = np.array(image_fpaths)
        
    
    def check_dataset(self):
        print('━━━━━━──────┄┄┄┄┄┄┈┈┈┈┈┈')
        print('Checking dataset...')
        checklist = {
            'data_path_exist': CheckStatus(),
            'data_dir_exist': CheckStatus(),
            'data_file_exist': CheckStatus()
        }
        
        checklist['data_path_exist'].set_status(os.path.isdir(self.data_dpath))
        checklist['data_dir_exist'].set_status(all((os.path.isdir(os.path.join(self.data_dpath, c)) for c in self.CLASSES)))
        if checklist['data_dir_exist'].is_passed():
            checklist['data_file_exist'].set_status(all((len([fn for fn in os.listdir(os.path.join(self.data_dpath, c)) if os.path.splitext(fn)[-1] in self.IMG_EXTS]) > 0 for c in self.CLASSES[:2])))
        
        # printing
        max_strlen = max([len(ckk) for ckk in checklist.keys()])
        lineheads = ['-'] * len(checklist.keys())
        for linehead, (ckk, ckv) in zip(lineheads, checklist.items()):
            padded = ckk.zfill(max_strlen).replace('0', ' ')
            print(f'{linehead} {padded}: {ckv.get_check_icon()}')
        print('━━━━━━──────┄┄┄┄┄┄┈┈┈┈┈┈')
        
        return all(checklist.values())
    
    def get_labels(self):
        return self.__label_arr.copy()
    
    @property
    def attrs(self):
        return ['gender_female']
    
    def idxof_attr(self, attr:str):
        if attr in self.attrs:
            return self.attrs.index(attr)
        else:
            return -1
    
    def __getitem__(self, index):
        image_fpath, gt_label = self.__imagefp_arr[index], self.__label_arr[index]
        image_fname = os.path.split(image_fpath)[-1]
        if os.path.isdir(image_fpath):
            frame_list = os.listdir(image_fpath)
            selected_frame_name = random.choice(frame_list)
            img = Image.open(os.path.join(image_fpath, selected_frame_name))
        elif os.path.isfile(image_fpath):
            img = Image.open(image_fpath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform:
            gt_label = gt_label[self.target_transform]
        return img, gt_label, image_fname
    
    def __len__(self):
        return self.__label_arr.shape[0]



class Pa100kDataset(ParDataset):
    def __init__(self):
        super().__init__()
    
    def get_labels(self):
        pass
    
    def get_n_attrs(self):
        pass



@PEDES_DATASET.register('adddipar')
def adddi_par_dataset(*args, **kwargs):
    return AdddiParDataset(*args, **kwargs)

@PEDES_DATASET.register('pa100k')
def pa100k_dataset(*args, **kwargs):
    return Pa100kDataset(*args, **kwargs)

@PEDES_DATASET.register('adddigencls')
def adddi_gencls_dataset(*args, **kwargs):
    return AdddiGenClsTestDataset(*args, **kwargs)

if __name__ == '__main__':
    from configs import cfg, update_config
    update_config(cfg, './configs/adddi.yaml')
    dataset = AdddiParDataset(cfg, split='test')
    print(dataset.attrs)
    print(dataset[0])
    