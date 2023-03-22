import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import glob

class NonDistortedAugmentation:
    def __init__(self, image_size, mean, std):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(size=self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img):
        return self.transform(img)

class FASDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, is_train=True, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transform or NonDistortedAugmentation((224, 224), self.mean, self.std)
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        img_name = os.path.join(self.root_dir, "images", img_name)
        
        img = Image.open(img_name)
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        
        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2, label

    def __len__(self):
        return len(self.data)
