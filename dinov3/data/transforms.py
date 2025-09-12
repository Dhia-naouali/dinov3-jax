# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3


"""
this file was imported as is from the original repo since no changes are needed to adapt to flax/jax
"""

import logging

import numpy as np
from torch import nn
from torchvision import transforms

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = transforms.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = transforms.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = transforms.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output












































































# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
from typing import Sequence

import PIL
import torch
from torchvision import transforms

logger = logging.getLogger("dinov3")


def make_interpolation_mode(mode_str: str) -> transforms.InterpolationMode:
    return {mode.value: mode for mode in transforms.InterpolationMode}[mode_str]


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Compose(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = transforms.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


class _MaxSizeResize(object):
    def __init__(
        self,
        max_size: int,
        interpolation: transforms.InterpolationMode,
    ):
        self._size = self._make_size(max_size)
        self._resampling = self._make_resampling(interpolation)

    def _make_size(self, max_size: int):
        return (max_size, max_size)

    def _make_resampling(self, interpolation: transforms.InterpolationMode):
        if interpolation == transforms.InterpolationMode.BICUBIC:
            return PIL.Image.Resampling.BICUBIC
        if interpolation == transforms.InterpolationMode.BILINEAR:
            return PIL.Image.Resampling.BILINEAR
        assert interpolation == transforms.InterpolationMode.NEAREST
        return PIL.Image.Resampling.NEAREST

    def __call__(self, image):
        image.thumbnail(size=self._size, resample=self._resampling)
        return image


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = transforms.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = _MaxSizeResize(max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = transforms.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = []
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(transforms.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = transforms.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


class MultipleResize(object):
    # A resize transform that makes the large side a multiple of a given number. That might change the aspect ratio.
    def __init__(self, interpolation=transforms.InterpolationMode.BILINEAR, multiple=1):
        self.multiple = multiple
        self.interpolation = interpolation

    def __call__(self, img):
        if self.multiple == 1:
            return img
        if hasattr(img, "shape"):
            h, w = img.shape[-2:]
        else:
            assert isinstance(
                img, PIL.Image.Image
            ), f"img should have a `shape` attribute or be a PIL Image, got {type(img)}"
            w, h = img.size
        new_h, new_w = [math.ceil(s / self.multiple) * self.multiple for s in (h, w)]
        resized_image = transforms.functional.resize(img, (new_h, new_w))
        return resized_image


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None