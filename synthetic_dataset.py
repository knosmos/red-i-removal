from typing import List
import os
import random

from PIL import Image, ImageFont, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


class SyntheticDataset(Dataset):
    """
    Dataset for generating on-the-fly synthetic images with overlaid red 'i' characters.

    Parameters
    ----------
    data_fname : str
        Path to the .npz file containing the base images. We expect ImageNet format (N x 12288) inside a 'data' key.
    transform : torch.nn.Module, optional
        Optional transform to be applied on a sample. Default is None.
    """
    def __init__(self, data_fname, transform: torch.nn.Module = None) -> None:
        self.data_fname = data_fname
        self.transform = transform
        self.fonts = self.load_fonts()
        self.data = np.load(data_fname, allow_pickle=True)["data"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        image = self.load_image(idx)

        if self.transform:
            image = self.transform(image)

        return image

    def load_fonts(self) -> List:
        fonts = []
        for name in os.listdir("fonts"):
            try:
                font = ImageFont.truetype(os.path.join("fonts", name), size=32)
                fonts.append(font)
            except IOError:
                continue
        print(f"Loaded {len(fonts)} fonts.")
        return fonts

    def load_image(self, idx) -> torch.Tensor:
        # load
        img = self.data[idx].reshape(3, 64, 64)
        img = np.transpose(img, (1, 2, 0))
        image = Image.fromarray(img)
        image = image.convert("RGB")
        # image = image.resize((256, 256))
        gt = image.copy()

        # blit a red i
        draw = ImageDraw.Draw(image)
        font = random.choice(self.fonts)
        x, y = random.randint(0, image.width - 32), random.randint(0, image.height - 32)
        draw.text((x, y), "i", font=font, fill=(255, 0, 0))

        # to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(np.array(gt)).permute(2, 0, 1).float() / 255.0
        return {"input": image, "gt": gt}


if __name__ == "__main__":
    dataset = SyntheticDataset("train_data_batch_1")
    sample = dataset[3]
    print(sample["input"].shape, sample["gt"].shape)
    cv2.imshow(
        "input",
        cv2.cvtColor(
            (sample["input"] * 255).permute(1, 2, 0).numpy().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )
    cv2.imshow(
        "gt",
        cv2.cvtColor(
            (sample["gt"] * 255).permute(1, 2, 0).numpy().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )
    cv2.waitKey(0)
