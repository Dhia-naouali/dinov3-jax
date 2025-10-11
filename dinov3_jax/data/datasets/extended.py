# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


# this file was imported as is from the original repo since no changes are needed to adapt to flax/jax


from typing import Any, Tuple
from torchvision.datasets import VisionDataset

from .decoders import Decoder, ImageDataDecoder, TargetDecoder


class ExtendedVisionDataset(VisionDataset):
    def __init__(
        self,
        image_decoder: Decoder = ImageDataDecoder,
        target_decoder: Decoder = TargetDecoder,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.image_decoder = image_decoder
        self.target_decoder = target_decoder

    def get_image_data(self, index: int) -> bytes:
        raise NotImplementedError

    def get_target(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = self.image_decoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
        return {
            "image": image.numpy().transpose(1, 2, 0), # HWC 
            "target": target.numpy()
        }

    def __len__(self) -> int:
        raise NotImplementedError