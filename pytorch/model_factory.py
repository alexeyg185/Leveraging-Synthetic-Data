from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.densenet import densenet121, densenet169, densenet161, densenet201
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
from PIL import Image


def create_model(model_name, num_classes=1000, pretrained=False, **kwargs):
    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = True

    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif model_name == 'inception_v3':
        model = inception_v3(
            pretrained=pretrained, num_classes=num_classes, transform_input=False, **kwargs)
    else:
        assert False, "Unknown model architecture (%s)" % model_name
    return model


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


DEFAULT_CROP_PCT = 0.875


def get_transforms_eval(model_name, img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if 'dpn' in model_name:
        if crop_pct is None:
            # Use default 87.5% crop for model's native img_size
            # but use 100% crop for larger than native as it
            # improves test time results across all models.
            if img_size == 224:
                scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))
            else:
                scale_size = img_size
        else:
            scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[124 / 255, 117 / 255, 104 / 255],
            std=[1 / (.0167 * 255)] * 3)
    elif 'inception' in model_name:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = LeNormalize()
    else:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize])
