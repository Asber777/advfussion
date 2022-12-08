import pandas as pd
import numpy as np
from PIL import Image 
from torch.utils.data.dataset import Dataset
from robustbench.data import PREPROCESSINGS
from torchvision import transforms
import os
PREPROCESSINGS['Res256Crop256'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
PREPROCESSINGS['Res64Crop64'] = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

class MyCustomDataset(Dataset):
    def __init__(self, img_path = "images", transform = 'Res256Crop256'):
        # Preprocess
        if transform in PREPROCESSINGS:
            self.transform = PREPROCESSINGS[transform]
        else:
            raise ValueError("transform must be 'Res256Crop256' or 'Res64Crop64'")
        self.image_name = np.asarray(os.listdir(img_path))
        self.data_len = self.image_name.shape[0]
        self.img_path = img_path

    def __getitem__(self, index):
        single_image_name = self.image_name[index]
        img_as_img = Image.open(os.path.join(self.img_path, single_image_name)) 
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        single_image_label = int(single_image_name.split('.')[0])
        return (img_as_tensor, single_image_label, single_image_name)

    def __len__(self):
        return self.data_len

class OnePicDataset(Dataset):
    def __init__(self, img_path = "images", image_name = '44.png', transform = 'Res256Crop256'):
        # Preprocess
        if transform in PREPROCESSINGS:
            self.transform = PREPROCESSINGS[transform]
        else:
            raise ValueError("transform must be 'Res256Crop256' or 'Res64Crop64'")
        self.data_len = 1
        self.image_name = image_name
        self.img_path = img_path

    def __getitem__(self, index):
        img_as_img = Image.open(os.path.join(self.img_path, self.image_name)) 
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        single_image_label = int(self.image_name.split('.')[0])
        return (img_as_tensor, single_image_label, self.image_name)

    def __len__(self):
        return self.data_len