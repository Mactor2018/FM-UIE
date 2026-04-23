"""Paired UIE Dataset for underwater image enhancement with Flow Matching."""

import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairedUIEDataset(Dataset):
    """Dataset that loads paired (degraded, ground-truth) underwater images.

    Args:
        ud_dir: Path to the directory containing degraded (underwater) images.
        gt_dir: Path to the directory containing ground-truth (enhanced) images.
        augment: Whether to apply synchronized geometric augmentations.
    """

    def __init__(self, ud_dir: str, gt_dir: str, augment: bool = False):
        self.ud_dir = ud_dir
        self.gt_dir = gt_dir
        self.augment = augment

        # Collect image filenames (support .png, .jpg, .jpeg)
        self.filenames = sorted([
            f for f in os.listdir(ud_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Verify every ud image has a matching gt image
        for f in self.filenames:
            gt_path = os.path.join(gt_dir, f)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"GT file not found: {gt_path}")

        print(f"  [PairedUIEDataset] Loaded {len(self.filenames)} pairs "
              f"from {ud_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        ud_img = Image.open(os.path.join(self.ud_dir, fname)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, fname)).convert('RGB')

        # Synchronized geometric augmentation (ud and gt stay aligned)
        if self.augment:
            ud_img, gt_img = self._sync_augment(ud_img, gt_img)

        # To tensor [0, 1] then normalize to [-1, 1]
        ud_tensor = TF.to_tensor(ud_img) * 2.0 - 1.0
        gt_tensor = TF.to_tensor(gt_img) * 2.0 - 1.0

        return ud_tensor, gt_tensor

    @staticmethod
    def _sync_augment(img1, img2):
        """Apply identical geometric transforms to both images.

        Only geometric augmentations are used — NO color/brightness jitter,
        which would corrupt the UIE colour mapping relationship.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # Random vertical flip
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # Random 90-degree rotation (lossless via PIL transpose)
        k = random.randint(0, 3)
        if k == 1:
            img1 = img1.transpose(Image.ROTATE_90)
            img2 = img2.transpose(Image.ROTATE_90)
        elif k == 2:
            img1 = img1.transpose(Image.ROTATE_180)
            img2 = img2.transpose(Image.ROTATE_180)
        elif k == 3:
            img1 = img1.transpose(Image.ROTATE_270)
            img2 = img2.transpose(Image.ROTATE_270)

        return img1, img2
