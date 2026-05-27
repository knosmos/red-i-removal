from typing import List
import random

from PIL import Image, ImageFont, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, image_paths: List[str], transform: torch.nn.Module=None) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.fonts = self.load_fonts()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def load_fonts(self) -> List:
        fonts = []
        for name in ["Arial.ttf", "Verdana.ttf", "TimesNewRoman.ttf"]:
            try:
                font = ImageFont.truetype(name, size=32)
                fonts.append(font)
            except IOError:
                continue
        return fonts

    def load_image(self, path) -> torch.Tensor:
        # load
        image = Image.open(path).convert('RGB')
        gt = image.copy()

        # blit a red i
        draw = ImageDraw.Draw(image)
        font = random.choice(self.fonts)
        x, y = random.randint(0, image.width - 32), random.randint(0, image.height - 32)
        draw.text((x, y), "i", font=font, fill=(255, 0, 0))
        
        # to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(np.array(gt)).permute(2, 0, 1).float() / 255.0
        return {'input': image, 'gt': gt}