import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def get_no_bg_transform() -> list:
    return [
        transforms.RandomAffine(degrees=(0, 180), translate=(1 / 3, 1 / 3), scale=(0.1, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]

def get_compose_transform(do_augmentation: bool) -> transforms.Compose:
    transform = []
    if do_augmentation:
        transform.extend(
            get_no_bg_transform()
        )
    transform.extend(
        [transforms.Resize(size=[224, 224]),
         transforms.ToTensor()]
    )
    return transforms.Compose(transform)

def get_imagnet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def get_imagnet_transform_test() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size=[224, 224], interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.497, 0.459, 0.415],
        #                      std=[0.302, 0.299, 0.309])
    ])