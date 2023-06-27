# from tkinter import Image
from typing import Tuple, Any, Optional, Callable

import torch
import torchvision.datasets as datasets

import numpy as np
import matplotlib
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional

from my_utils.background_picker import BackgroundRandomPicker
from my_utils.backgrounds import is_background_good
from my_utils.class_to_backgrounds import class_to_backgrounds
from my_utils.image_tools import perform_images_composition
from my_utils.imagenet_to_synth import inverse_imgnet_to_synt
from PIL import Image

from pytorch.custom_transforms import get_compose_transform, get_imagnet_transform, get_imagnet_transform_test

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class SyntheticDataset(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            include_background: int = -1,
            test=False
    ):
        super(SyntheticDataset, self).__init__(root=root, loader=loader,
                                               transform=transform,
                                               target_transform=target_transform,
                                               is_valid_file=is_valid_file)
        bg_path = "/home/alexey.gruzdev/Documents/bench_project/backgrounds/places2/train_256_places365standard/flat"
        self.bg_picker = BackgroundRandomPicker(imagnet_cls_2_background_map=class_to_backgrounds, backgrounds_root_path=bg_path)
        self.synth_to_imagnet_map = inverse_imgnet_to_synt()
        self.include_background = include_background
        self.test = test

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        fg_path, target = self.samples[index]
        fg_path_split = fg_path.split(sep='/')

        if self.test:
            is_imagenet = 'ILSVR' in fg_path_split[-1]
            sample = self.loader(fg_path)
            if is_imagenet:
                dyn_transforms = get_imagnet_transform_test()
            else:
                dyn_transforms = get_compose_transform(do_augmentation=False)
        else:
            is_imagenet = '_imagenet' in fg_path_split[-2]
            if is_imagenet:
                incl_bg = False
            elif self.include_background == -1:
                incl_bg = list(is_background_good.values())[target]
            elif self.include_background == 1:
                incl_bg = True
            elif self.include_background == 0:
                incl_bg = False
            else:
                raise RuntimeError()

            if is_imagenet:
                sample = self.loader(fg_path)
                dyn_transforms = get_imagnet_transform()

            elif incl_bg:
                kls = self.classes[target]
                bg_path = self.bg_picker.pick_random_background_by_cls(imagnet_cls=self.synth_to_imagnet_map[kls])

                bg = Image.open(bg_path)
                fg = Image.open(fg_path)
                sample = perform_images_composition(bg=bg, fg=fg)
                dyn_transforms = get_compose_transform(do_augmentation=True)

            else:
                sample = self.loader(fg_path)
                dyn_transforms = get_compose_transform(do_augmentation=False)

        if dyn_transforms is not None:
            sample = dyn_transforms(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # show_image(np_img=sample.cpu().detach().numpy(), title=str(self.classes[target]))
        # save_image(np_img=sample.cpu().detach().numpy(), cls_id=target, img_id=index)
        return sample, target, fg_path

    def filenames(self):
        return [x[0] for x in self.imgs]

def show_image(np_img, title):
    plt.title(title)
    plt.imshow(np.moveaxis(np_img, 0, 2))
    plt.show()

def save_image(np_img, cls_id, img_id):
    plt.imsave(fname=f"/home/alexey.gruzdev/Documents/bench_project/synth_data/samples/tests/{cls_id}_{img_id}.png", arr=np.moveaxis(np_img, 0, 2))


