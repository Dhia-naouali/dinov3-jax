# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


# this file was imported as is from the original repo since no changes are needed to adapt to flax/jax

import json
import os
import random
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

# Dataset: https://www.kaggle.com/datasets/nikhil7280/coco-image-caption


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"


def read_images_and_captions(root: str, split: _Split) -> List[Dict]:
    image_dir = None
    if _Split(split) == _Split.TRAIN:
        annotations_full_path = os.path.join(
            root, "annotations_trainval2014/annotations/captions_train2014.json"
        )
        image_dir = os.path.join(root, "train2014/train2014")
    else:
        annotations_full_path = os.path.join(
            root, "annotations_trainval2017/annotations/captions_train2017.json"
        )
        image_dir = os.path.join(root, "val2017/val2017")
    with open(annotations_full_path) as f:
        all_annotations = json.load(f)
    data = {}
    for item in all_annotations["images"]:
        id = item["id"]
        data[id] = {
            "id": None,
            "image": os.path.join(image_dir, item["file_name"]),
            "captions": [],
        }
    for item in all_annotations["annotations"]:
        data[item["image_id"]]["id"] = item["image_id"]
        data[item["image_id"]]["captions"].append(item["caption"])
    return list(data.values())


class CocoCaptions(ExtendedVisionDataset):
    Split = Union[_Split]


    def __getitem__(self, idx):
        # bypassing i/o
        image = np.random.randn(224, 224, 3)
        target = np.random.randint((1,), 1000)
        return {
            "image": image,
            "target": target
        }

    def __init__(
        self,
        *,
        split: "CocoCaptions.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        return None
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self.image_captions = read_images_and_captions(root, split)

    def get_image_relpath(self, index: int) -> str:
        image_path = self.image_captions[index]["image"]
        return image_path

    def get_image_data(self, index: int) -> bytes:
        image_path = self.get_image_relpath(index)
        with open(image_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> str:
        return random.choice(self.image_captions[index]["captions"])

    def __len__(self) -> int:
        return 1000
        return len(self.image_captions)