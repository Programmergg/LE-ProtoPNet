
import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import Sampler
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random
import dask.dataframe as dd
import pytorch_lightning as pl
from collections import defaultdict

def load_attributes(path_to_file, num_images=11788, num_attributes=312):
    attributes = dd.read_csv(path_to_file, sep=r'\s+', header=None, usecols=[0, 1, 2, 3, 4])
    attributes = attributes.compute() 
    attr_matrix = np.zeros((num_images, num_attributes), dtype=np.int64)
    for _, row in attributes.iterrows():
        image_id = int(row[0]) - 1
        attribute_id = int(row[1]) - 1
        is_present = row[2]
        if is_present == 1 or is_present == 2 or is_present == 3:
            attr_matrix[image_id, attribute_id] = 1
    attr_tensor = torch.tensor(attr_matrix)
    return attr_tensor


def get_transform(augmentation_severity=1):
    rotation_range = augmentation_severity * 5  
    color_jitter_factor = augmentation_severity * 0.1  
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.AugMix(severity=augmentation_severity),
        transforms.ToTensor(),
    ])
    return transform

class CUBDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_list=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_images = {}
        with open(os.path.join(data_dir, "images.txt")) as f:
            images = [line.strip().split() for line in f]
        with open(os.path.join(data_dir, "image_class_labels.txt")) as f:
            labels = [int(line.strip().split()[1]) for line in f]

        for img_info, label in zip(images, labels):
            if label in class_list:
                self.image_paths.append(os.path.join(data_dir, "images", img_info[1]))
                self.labels.append(label)
                if label not in self.class_to_images:
                    self.class_to_images[label] = []
                self.class_to_images[label].append(len(self.labels) - 1)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
       
        return image, label

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
        for _ in range(self.n_iter):  
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

def create_meta_dataloader(data_dir, class_list, n_way, k_shot, q_query,n_iter):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = CUBDataset(data_dir, transform=transform, class_list=class_list)
    batch_sampler = MetaBatchSampler(dataset.class_to_images, n_way, k_shot, q_query,n_iter)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    return dataloader

class MetaLearningDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir, n_way, k_shot, q_query, n_iter,transform=get_transform(augmentation_severity=8)):
        super().__init__()
        self.data_dir = data_dir
        self.train_classes, self.test_classes = split_cub_classes(num_classes=200, train_ratio=0.8)
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
        
        self.train_dataset = CUBDataset(self.data_dir, transform=self.transform, class_list=self.train_classes)
        self.test_dataset = CUBDataset(self.data_dir, transform=test_transform, class_list=self.test_classes)

    def train_dataloader(self):
        batch_sampler_train = MetaBatchSampler(self.train_dataset.class_to_images, self.n_way, self.k_shot, self.q_query,self.n_iter)
        train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler_train,num_workers=4)
        return train_loader 

    def test_dataloader(self):   
        batch_sampler_test = MetaBatchSampler(self.test_dataset.class_to_images, self.n_way, self.k_shot, self.q_query,self.n_iter)
        test_loader = DataLoader(self.test_dataset,  batch_sampler=batch_sampler_test,num_workers=4)
        return test_loader 
    
class CUBdataloader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128,transform=get_transform(augmentation_severity=8)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_classes, self.test_classes = split_cub_classes(num_classes=200, train_ratio=0.8)
        self.transform = transform
        self.dims =  (3, 224, 224)
        self.weight_class = None
        self.std = [0.229, 0.224, 0.225]
        self.mean =[0.485, 0.456, 0.406]
        self.num_classes = 200
        self.img_size = 224
        self.augmentation_severity = 8
    #def setup(self, stage=None):
        self.train_dataset = CUBDataset(self.data_dir, transform=self.transform, class_list=self.train_classes)
        self.test_dataset = CUBDataset(self.data_dir, transform=self.transform, class_list=self.test_classes)
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return train_loader
    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return test_loader

def split_cub_classes(num_classes, train_ratio):
    indices = np.arange(num_classes)
    train_classes, test_classes = train_test_split(indices, train_size=train_ratio, random_state=42)
    return train_classes, test_classes


if __name__ == '__main__':
    data_dir =' '
    few_shot_loader = MetaLearningDataLoader(data_dir, n_way=5, k_shot=5, q_query=15,n_iter=100)
    few_shot_loader.setup()
    train_dataset = few_shot_loader.train_dataset
    test_dataset = few_shot_loader.test_dataset
    test_loader = few_shot_loader.test_dataloader()

    
    for batch in test_loader:
        images, labels = batch
        print("test")
        print(images.shape, labels)
        