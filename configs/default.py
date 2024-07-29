# from yacs.config import CfgNode as CN
import re
import typing
from triton import Config
import yaml  # PyYaml


class ConfigNamespace:
    def __init__(self, *args, **kwargs):
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and not callable(v):
                self.update({k: v})
        for k in kwargs.keys():
            self.update({k: kwargs[k]})
        self.make_recursive_namespace()

    def make_recursive_namespace(self, target=None):
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                new_name_space = ConfigNamespace(**v)
                self.__dict__.update({k: new_name_space})

    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def update(self, d):
        self.__dict__.update(d)
    
    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        txt = '\n'.join(f'{k}: {v}' for k, v in vars(self).items())
        return txt

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)
    

class BasicCN(ConfigNamespace):
    NAME = "default"
    DISTRIBUTTED = False
    
    class ReloadCN(ConfigNamespace):
        TYPE = False
        NAME = 'backbone'
        PTH = ''
    RELOAD = ReloadCN()
    
    class DatasetCN(ConfigNamespace):
        NAME = "PA100k"
        TARGETTRANSFORM = []
        PATH = ''
        ZERO_SHOT = False
        LABEL = 'eval'  # train on all labels, test on part labels (35 for peta, 51 for rap)
        TRAIN_SPLIT = 'trainval'
        VAL_SPLIT = 'val'
        TEST_SPLIT = 'test'
        HEIGHT = 256
        WIDTH = 192
    DATASET = DatasetCN()
    
    class BackboneCN(ConfigNamespace):
        TYPE = "resnet50"
        MULTISCALE = False
    BACKBONE = BackboneCN()
    
    class ClassifierCN(ConfigNamespace):
        TYPE = "base"
        NAME = ""
        POOLING = "avg"
        BN = False
        SCALE = 1
    CLASSIFIER = ClassifierCN()
    
    class TrainCN(ConfigNamespace):
        BATCH_SIZE = 64
        MAX_EPOCH = 30
        SHUFFLE = True
        NUM_WORKERS = 4
        CLIP_GRAD = False
        BN_WD = True
        AUX_LOSS_START = -1
        
        class DataAugCN(ConfigNamespace):
            TYPE = 'base'
            AUTOAUG_PROB = 0.5
        DATAAUG = DataAugCN()
        
        class EmaCN(ConfigNamespace):
            ENABLE = False
            DECAY = 0.9998
            FORCE_CPU = False
        EMA = EmaCN()
        
        class OptimizerCN(ConfigNamespace):
            TYPE = "SGD"
            MOMENTUM = 0.9
            WEIGHT_DECAY = 1e-4
        OPTIMIZER = OptimizerCN()
        
        class LrSchedulerCN(ConfigNamespace):
            TYPE = "plateau"
            LR_STEP = [0,]
            LR_FT = 1e-2
            LR_NEW = 1e-2
            WMUP_COEF = 0.01
            WMUP_LR_INIT = 1e-6
        LR_SCHEDULER = LrSchedulerCN()
        
    TRAIN = TrainCN()
    
    class InferCN(ConfigNamespace):
        SAMPLING = False
    INFER = InferCN()
    
    class LossCN(ConfigNamespace):
        TYPE = "bce"
        SAMPLE_WEIGHT = ""  # None
        LOSS_WEIGHT = [1, ]
        SIZESUM = True   # for a sample, BCE losses is the summation of all label instead of the average.
    LOSS = LossCN()
    
    class MetricCN(ConfigNamespace):
        TYPE = 'pedestrian'
    METRIC = MetricCN()
    
    class VisCN(ConfigNamespace):
        CAM = 'valid'
    VIS = VisCN()
    
    class TransCN(ConfigNamespace):
        DIM_HIDDEN = 256
        DROPOUT = 0.1
        NHEADS = 8
        DIM_FFD = 2048
        ENC_LAYERS = 6
        DEC_LAYERS = 6
        PRE_NORM = False
        EOS_COEF = 0.1
        NUM_QUERIES = 100
    TRANS = TransCN()


_C = BasicCN()

def __yaml_load(file='data.yaml', append_filename=False):
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def __overrides_config(lower_cfg:ConfigNamespace, upper_cfg:ConfigNamespace):
    for cfgkey in upper_cfg.__dict__.keys():
        if isinstance(upper_cfg.__dict__[cfgkey], ConfigNamespace):
            __overrides_config(lower_cfg.__dict__[cfgkey], upper_cfg.__dict__[cfgkey])
        else:
            lower_cfg.__dict__.update({cfgkey: upper_cfg.__dict__[cfgkey]})

def update_config(cfg, cfg_fpath):
    cfg_dict = __yaml_load(cfg_fpath)
    if cfg_dict is None:
        n_cfg = ConfigNamespace(**{})
    else:
        for k, v in cfg_dict.items():
            if isinstance(v, str) and v.lower() == 'none':
                cfg_dict[k] = None
        n_cfg = ConfigNamespace(**cfg_dict)
    __overrides_config(cfg, n_cfg)
