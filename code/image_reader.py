import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from tqdm import tqdm
import os
import re

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
IMG_SIZE = 512
img_dir = "../data_generation/data/data_0/only_keys_detection"

def mask_to_closest_one_hot(mask, unique_values):
    mask_array = np.array(mask, dtype=np.float32)  # Shape: (IMAGE_SIZE, IMAGE_SIZE)
    mask_tensor = torch.tensor(mask_array).unsqueeze(0)  # Shape: (1, IMAGE_SIZE, IMAGE_SIZE)
    unique_tensor = torch.tensor(unique_values, dtype=torch.float32).view(-1, 1, 1)  # Shape: (3, 1, 1)
    distances = torch.abs(mask_tensor - unique_tensor)  # Shape: (3, IMAGE_SIZE, IMAGE_SIZE)
    closest_indices = torch.argmin(distances, dim=0)  # Shape: (IMAGE_SIZE, IMAGE_SIZE)
    one_hot = torch.nn.functional.one_hot(closest_indices, num_classes=len(unique_values))  # Shape: (IMAGE_SIZE, IMAGE_SIZE, 3)
    one_hot = one_hot.permute(2, 0, 1).float()

    return one_hot


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        all_images = os.listdir(root_dir)
        self.images = [img for img in all_images if re.search("Image", img)]
        print(f"Found {len(self.images)} images")
        self.items = []
        loop = tqdm(self.images)
        for img_path in loop:
            image =  Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")
            mask = Image.open(os.path.join(self.root_dir, self._get_mask_for_image(img_path))).convert("L")
            unique_values = np.unique(mask)
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            transform2 = Compose([ToTensor()])
            mask = mask_to_closest_one_hot(mask, unique_values)
            
            image = transform2(image)
            self.items.append((image, mask))


    def __len__(self):
        return len(self.images)

    def _get_mask_for_image(self, img):
        return img.replace("Image", "Segmentation")

    def __getitem__(self, idx):
        return self.items[idx]
    
transforms = Compose([
    CenterCrop(1080),
    Resize((IMG_SIZE, IMG_SIZE)),
])

def get_images(image_dir=img_dir, transform = transforms, batch_size = 8, shuffle = True, pin_memory = True):
    dataset = ImageDataset(image_dir, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_loader, test_loader