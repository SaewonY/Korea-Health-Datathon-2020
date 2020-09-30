import math
import random
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation


def get_transform(
        target_size=(512, 512),
        transform_list='horizontal_flip', # random_crop | keep_aspect
        augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    transform_list = transform_list.split(', ')
    augments = list()

    for transform_name in transform_list:
        if transform_name == 'random_crop':
            scale = (0.6, 1.0) if is_train else (0.8, 1.0)
            transform.append(RandomResizedCrop(target_size, scale=scale))
        # elif transform_name == 'resize':
        #     transform.append(Resize(target_size))
        elif transform_name == 'keep_aspect':
            transform.append(KeepAsepctResize(target_size))
        elif transform_name == 'Affine':
            augments.append(RandomAffine(degrees=(-180, 180),
                                         scale=(0.8889, 1.0),
                                         shear=(-36, 36)))
        elif transform_name == 'centor_crop':
            augments.append(CenterCrop(target_size))
        elif transform_name == 'horizontal_flip':
            augments.append(RandomHorizontalFlip())
        elif transform_name == 'vertical_flip':
            augments.append(RandomVerticalFlip())
        elif transform == 'random_grayscale':
            p = 0.5 if is_train else 0.25
            transform.append(RandomGrayscale(p))
        elif transform_name == 'random_rotate':
            augments.append(RandomRotation(180))
        elif transform_name == 'color_jitter':
            brightness = 0.1 if is_train else 0.05
            contrast = 0.1 if is_train else 0.05
            augments.append(ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0,
                hue=0,
                ))

    transform.append(RandomApply(augments, p=augment_ratio))   
    transform.append(ToTensor())
    transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transform)
