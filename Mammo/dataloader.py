import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 

def path_loader(root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader(root_path, keys):
    labels_dict = {}
    labels = []
    with open(os.path.join(root_path, 'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels


class PathDataset(Dataset): 
    def __init__(self, image_paths, labels=None, default_transforms=None, transforms=None, is_test=False): 
        self.image_paths = image_paths
        self.labels = labels 
        self.default_transforms = default_transforms
        self.transforms = transforms
        self.is_test = is_test

        self.imgs = []

        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            if self.default_transforms is not None:
                img = self.default_transforms(img)
    
            self.imgs.append(img)

    def __getitem__(self, index):
        
        img = self.imgs[index]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.is_test:
            return torch.tensor(img, dtype=torch.float32)
        else:
            return torch.tensor(img, dtype=torch.float32),\
                 torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self): 
        return len(self.image_paths)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img