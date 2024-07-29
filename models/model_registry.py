from .backbone.swin_transformer import SwinTransformer
from .backbone.resnet import _resnet, BasicBlock, Bottleneck

from ops.losses.bceloss import BCELoss
from ops.losses.scaledbceloss import ScaledBCELoss

from tools.registry import Registry

BACKBONE = Registry()
CLASSIFIER = Registry()
LOSSES = Registry()


@BACKBONE.register("swin_b")
def swin_base_patch4_window7_224(pretrained='./pretrained/solider_swin_base.pth', convert_weights=False, img_size=224,drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0., **kwargs):
    model = SwinTransformer(pretrained=pretrained, convert_weights=convert_weights, pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    return model

@BACKBONE.register("swin_s")
def swin_small_patch4_window7_224(pretrained='./pretrained/solider_swin_small.pth', convert_weights=False, img_size=224,drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0., **kwargs):
    model = SwinTransformer(pretrained=pretrained, convert_weights=convert_weights, pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    return model

@BACKBONE.register("swin_t")
def swin_tiny_patch4_window7_224(pretrained='./pretrained/solider_swin_tiny.pth', convert_weights=False, img_size=224,drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0., **kwargs):
    model = SwinTransformer(pretrained=pretrained, convert_weights=convert_weights, pretrain_img_size = img_size, patch_size=4, window_size=7, embed_dims=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, **kwargs)
    return model

@BACKBONE.register("resnet18")
def resnet18(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

@BACKBONE.register("resnet34")
def resnet34(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

@BACKBONE.register("resnet50")
def resnet50(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

@BACKBONE.register("resnet101")
def resnet101(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

@BACKBONE.register("resnet152")
def resnet152(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

@BACKBONE.register("resnext50_32x4d")
def resnext50_32x4d(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

@BACKBONE.register("resnext101_32x8d")
def resnext101_32x8d(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

@LOSSES.register("bceloss")
def bce_loss(*args, **kwargs):
    return BCELoss(*args, **kwargs)

@LOSSES.register("scaledbceloss")
def scaled_bce_loss(*args, **kwargs):
    return ScaledBCELoss(*args, **kwargs)