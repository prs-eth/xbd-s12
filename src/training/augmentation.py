"""Augmentation transforms for training."""

import random

import torch


class RandomFlipRotate90:
    """
    Randomly apply horizontal flip, vertical flip, and 90-degree rotations to a list of images.

    The images do not need to have the same size.
    """

    def __init__(self, p_h: float = 0.5, p_v: float = 0.5, p_r: float = 1.0):
        self.p_h = p_h
        self.p_v = p_v
        self.p_r = p_r

    def get_params(self) -> tuple[bool, bool, int]:
        # decide once
        do_h = random.random() < self.p_h
        do_v = random.random() < self.p_v
        k = random.randint(0, 3) if random.random() < self.p_r else 0
        return do_h, do_v, k

    def apply(self, img: torch.Tensor, do_h: bool, do_v: bool, k: int) -> torch.Tensor:
        if do_h:
            img = torch.flip(img, dims=[-1])  # horizontal
        if do_v:
            img = torch.flip(img, dims=[-2])  # vertical
        if k > 0:
            img = torch.rot90(img, k, dims=[-2, -1])  # 90° multiples
        return img

    def __call__(self, *imgs: torch.Tensor) -> list[torch.Tensor]:
        do_h, do_v, k = self.get_params()
        return [self.apply(img, do_h, do_v, k) for img in imgs]
