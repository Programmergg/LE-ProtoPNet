import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler
import numpy as np
import pytorch_lightning as pl
class aircraftDataset(Dataset):
    def __init__(self, root_dir, transform=None, N=5, K=5,Q=15, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.N = N  # N-way
        self.K = K  # K-shot
        self.Q = Q 
        self.is_train = is_train

        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.class_images = defaultdict(list)
        self.img_paths = []
        self.labels = {}
        for class_name in self.class_names:
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.class_images[class_name].append(img_path)
                    self.img_paths.append(img_path)
                    self.labels[img_path] = class_name
        
    def __len__(self):
        return 10000 
    def __getitem__(self, idx):
      #  print(idx)
        img_paths = idx
        img=Image.open(img_paths).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
      


class MetaBatchSampler(Sampler):
    def __init__(self, class_to_images, n_way, k_shot, q_query,n_iter):
        self.class_to_images = class_to_images
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_iter = n_iter
        self.classes = list(class_to_images.keys())
        self.image_counts = defaultdict(int)
    def __iter__(self):
        for _ in range(self.n_iter):  # 每个 epoch 生成 n_iter 个 batch
            support_indices = []
            query_indices = []
            sampled_classes = random.sample(self.classes, self.n_way)
        
            for cls in sampled_classes:
                indices = random.sample(self.class_to_images[cls], self.k_shot + self.q_query)
                support_indices.extend(indices[:self.k_shot])
                query_indices.extend(indices[self.k_shot:])

            for idx in indices:
                    self.image_counts[idx] += 1
                
            batch_indices = support_indices + query_indices
            yield batch_indices

    def __len__(self):
        return self.n_iter



class aircraft_DataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, n_way, k_shot, q_query, n_iter,transform):
        super().__init__()
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_iter = n_iter
        self.dims =  (3, 224, 224)
        self.weight_class = None
        self.std = [0.229, 0.224, 0.225]
        self.mean =[0.485, 0.456, 0.406]
        self.num_classes = 200
        self.img_size = 224
        self.augmentation_severity = 8
        self.transform = transform

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.6, 1.0), ratio=(0.7, 1.1)),
                transforms.AugMix(severity=self.augmentation_severity),
                transforms.ToTensor(),
                # DropBlock2D(drop_prob=0.25, block_size=25),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        
        self.train_dataset = aircraftDataset(root_dir=' ', transform=train_transform, N=self.n_way, K=self.k_shot, Q=self.q_query, is_train=True) 
        
        self.test_dataset = aircraftDataset(root_dir=' ', transform=test_transform, N=self.n_way, K=self.k_shot, Q=self.q_query, is_train=False) 
    def train_dataloader(self):
        batch_sampler_train = MetaBatchSampler(self.train_dataset.class_images, self.n_way, self.k_shot, self.q_query,self.n_iter)
        train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler_train,num_workers=4)
        return train_loader 
    
    def test_dataloader(self):   
        batch_sampler_test = MetaBatchSampler(self.test_dataset.class_images, self.n_way, self.k_shot, self.q_query,self.n_iter)
        test_loader = DataLoader(self.test_dataset,  batch_sampler=batch_sampler_test,num_workers=4)
        return test_loader 


if __name__ == '__main__':
    root_dir = ' '
    train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.6, 1.0), ratio=(0.7, 1.1)),
                transforms.AugMix(severity=8),
                transforms.ToTensor(),
                # DropBlock2D(drop_prob=0.25, block_size=25),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    train_dataset = aircraftDataset(root_dir, transform=train_transform, N=5, K=5, Q=15, is_train=True)
    print(train_dataset.class_names)
   # print(train_dataset.img_paths)
    loader = aircraft_DataLoader(data_dir=' ', n_way=5, k_shot=5, q_query=15, n_iter=10, transform=train_transform)
    loader.setup()
    #train_dataset = loader.train_dataset
    train_loader = loader.train_dataloader()
    for i, batch in enumerate(train_loader):
        img, label = batch
        print(img.shape, label)
        if i == 1:
            break