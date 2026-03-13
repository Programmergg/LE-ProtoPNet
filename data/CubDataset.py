import os
import pandas as pd
from torchvision import transforms
from typing import Any, Callable, Optional
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#  Adapted from: https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
import pandas as pd
import numpy as np
import torch


def load_attributes(path_to_file, num_images=11788, num_attributes=312):
    attributes = pd.read_csv(path_to_file, sep=r'\s+', header=None, engine='python',usecols=[0, 1, 2, 3, 4])
    attr_matrix = np.zeros((num_images, num_attributes), dtype=np.int64)
    for _, row in attributes.iterrows():
        image_id = int(row[0]) - 1  # 将 image_id 转换为整数并从1基准转换为0基准
        attribute_id = int(row[1]) - 1  # 将 attribute_id 转换为整数并从1基准转换为0基准
        is_present = row[2]
        if is_present == 1 or is_present == 2 or is_present == 3:  # 仅考虑存在的属性
            attr_matrix[image_id, attribute_id] = 1
    attr_tensor = torch.tensor(attr_matrix)
    
    return attr_tensor

class CategoriesSampler:
    def __init__(self, labels, n_iters, n_way, k_shot, query_per_class,train):
        """
        Args:
            labels (list or array): 标签列表
            n_way (int): 每个 episode 中的类别数
            k_shot (int): 每个类别的支持样本数
            query_per_class (int): 每个类别的查询样本数
            n_iters (int): 每个 epoch 中的 episode 数
        """
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.n_iters = n_iters
        self.classes = list(set(labels))
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.train = train

    def __iter__(self):
        if self.train:
            replace = False
        else:
            replace = True
        for _ in range(self.n_iters):
            selected_classes = np.random.choice(self.classes, self.n_way, replace=replace)
            support_indices = []
            query_indices = []
            for class_label in selected_classes:
                indices = np.random.choice(self.class_indices[class_label], self.k_shot + self.query_per_class, replace=replace)
                support_indices.extend(indices[:self.k_shot])
                query_indices.extend(indices[self.k_shot:])
            yield support_indices + query_indices

    def __len__(self):
        return self.n_iters


class CubDataset(VisionDataset):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(self, root: str, download: bool = False, train: bool = True, transform=None, target_transform=None, loader=default_loader, photometric_transforms=None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.photometric_transforms = photometric_transforms
        if not self._check_integrity() and not download:
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.data: Any = []
        self.targets = []
        if download:
            self.download()
        self._load_metadata()
        self.attributes = load_attributes('/home/WRS/ProtosViT/datasets/cub/CUB_200_2011/attributes/image_attribute_labels.txt')

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )
        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        attributes = self.attributes[idx] 
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.photometric_transforms is not None:
            img_aug = self.photometric_transforms(img)
            return img, target, img_aug ,attributes
        else:
            return img, target, attributes
        
class CubFewShotDataset(CubDataset):
    def __init__(self, root: str, download: bool = False, train: bool = True, transform=None, 
                 target_transform=None, loader=default_loader, photometric_transforms=None):
        super().__init__(root, download=download, train=train, transform=transform, 
                         target_transform=target_transform, loader=loader, photometric_transforms=photometric_transforms)
    
        self.labels = self.data['target'].values - 1  # 目标值从1开始，减去1以对齐到0

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, target, attributes = super().__getitem__(idx)
        return img, target, attributes
    
def get_transform(augmentation_severity=1):
    rotation_range = augmentation_severity * 5  # 每个severity等级增加5度
    color_jitter_factor = augmentation_severity * 0.1  # 每个severity增加0.1的抖动 
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(rotation_range),  # 旋转角度随着severity增加
        transforms.ColorJitter(brightness=color_jitter_factor,
                               contrast=color_jitter_factor,
                               saturation=color_jitter_factor,
                               hue=color_jitter_factor * 0.05),  # 随着severity调整颜色抖动
        transforms.ToTensor(),
    ])
    return transform

class CubFewShotDataModule(pl.LightningDataModule):
    def __init__(self, root, n_way=5, n_shot=5, n_query=15,  num_workers=4, n_iter=500, transform=get_transform(augmentation_severity=8)):
        super().__init__()
        self.root = root
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_workers = num_workers
        self.n_iter = n_iter
        self.transform = transform
        self.dims =  (3, 224, 224)
        self.std = [0.229, 0.224, 0.225]
        self.mean =[0.485, 0.456, 0.406]
        self.num_classes = 200
        self.weight_class = None

    def setup(self, stage=None):
        self.train_dataset = CubFewShotDataset(root=self.root, train=True, transform=self.transform)
        self.test_dataset = CubFewShotDataset(root=self.root, train=False, transform=self.transform)
        #self.train_dataset = FewShotDataset(root=self.root, set_name='CUB_200_2011',split='train', transform=self.transform)
        #self.test_dataset = FewShotDataset(root=self.root, set_name='CUB_200_2011',split='test', transform=self.transform)

    def train_dataloader(self):
        train_sampler = CategoriesSampler(self.train_dataset.labels, self.n_iter, self.n_way, self.n_shot, self.n_query,train=True)
        return DataLoader(self.train_dataset, batch_sampler=train_sampler, num_workers=self.num_workers)

    def test_dataloader(self):
        # 构建采样器
        test_sampler = CategoriesSampler(self.test_dataset.labels, self.n_iter, self.n_way, self.n_shot, self.n_query,train=False)

        # 构建 DataLoader
        return DataLoader(self.test_dataset, batch_sampler=test_sampler, num_workers=self.num_workers)
        

# import os
# import random
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image

# class CUBFewShotDataset(Dataset):
#     def __init__(self, root_dir, num_classes_per_task, num_support, num_query, transform=None, train= True):
#         self.root_dir = root_dir
#         self.num_classes_per_task = num_classes_per_task
#         self.num_support = num_support
#         self.num_query = num_query
#         self.transform = transform
#         self.train = train
#         self.class_to_images = self._load_class_to_images()
#     def _load_class_to_images(self):
#         image_file = os.path.join(self.root_dir, 'images.txt')
#         label_file = os.path.join(self.root_dir, 'image_class_labels.txt')
#         split_file = os.path.join(self.root_dir, 'train_test_split.txt')
#         image_df = pd.read_csv(image_file, sep=' ', header=None, names=['id', 'path'])
#         label_df = pd.read_csv(label_file, sep=' ', header=None, names=['id', 'label'])
#         split_df = pd.read_csv(split_file, sep=' ', header=None, names=['id', 'is_train'])
#         class_to_images = {}
#         for _, row in image_df.iterrows():
#             img_id = row['id']
#             img_path = os.path.join(self.root_dir, 'images', row['path'])
#             label = label_df[label_df['id'] == img_id]['label'].values[0]
#             is_train = split_df[split_df['id'] == img_id]['is_train'].values[0]

#             if self.train and is_train == 1:
#                 if label not in class_to_images:
#                     class_to_images[label] = []
#                 class_to_images[label].append(img_path)
#             elif not self.train and is_train == 0:
#                 if label not in class_to_images:
#                     class_to_images[label] = []
#                 class_to_images[label].append(img_path)

#         return class_to_images


#     def __len__(self):
#         return len(self.class_to_images)

#     def __getitem__(self, idx):
#         selected_classes = random.sample(list(self.class_to_images.keys()), self.num_classes_per_task)
#         support_set = []
#         query_set = []
#         for cls in selected_classes:
#             image_paths = self.class_to_images[cls]
#             random.shuffle(image_paths)
#             support_images = image_paths[:self.num_support]
#             query_images = image_paths[self.num_support:self.num_support + self.num_query]
#             support_set.extend([(self._load_image(p), cls) for p in support_images])
#             query_set.extend([(self._load_image(p), cls) for p in query_images])

#         return support_set, query_set

#     def _load_image(self, path):
#         img = Image.open(path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img

# import pytorch_lightning as pl
# import re
# class CUBFewShotDataModule(pl.LightningDataModule):
#     def __init__(self, root_dir, num_classes_per_task, num_support, num_query, batch_size,augmentation_severity=1) :
#         super().__init__()
#         self.root_dir = root_dir
#         self.num_classes_per_task = num_classes_per_task
#         self.num_support = num_support
#         self.num_query = num_query
#         self.batch_size = batch_size
#         self.augmentation_severity = augmentation_severity
#         self.transform = get_transform(augmentation_severity=self.augmentation_severity)

#     def setup(self, stage=None):
#         # all_images = self._load_images()
#         # train_images, test_images = self._split_train_test(all_images)
#         #train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)
#         train_transform = get_transform(augmentation_severity=self.augmentation_severity)
#         test_transform = get_transform(augmentation_severity=0)
#         self.train_dataset = CUBFewShotDataset(self.root_dir, self.num_classes_per_task, self.num_support, self.num_query, train_transform, train=True)
#         #self.val_dataset = CUBFewShotDataset(self.root_dir, self.num_classes_per_task, self.num_support, self.num_query, test_transform, val_images)
#         self.test_dataset = CUBFewShotDataset(self.root_dir, self.num_classes_per_task, self.num_support, self.num_query, test_transform, train=False)

#     # def _load_images(self):
#     #     image_file = os.path.join(self.root_dir, 'images.txt')
#     #     label_file = os.path.join(self.root_dir, 'image_class_labels.txt')
#     #     image_df = pd.read_csv(image_file, sep=' ', header=None, names=['id', 'path'])
#     #     label_df = pd.read_csv(label_file, sep=' ', header=None, names=['id', 'label'])
#     #     images = []
#     #     for _, row in image_df.iterrows():
#     #         img_id = row['id']
#     #         img_path = os.path.join(self.root_dir, 'images', row['path'])
#     #         label = label_df[label_df['id'] == img_id]['label'].values[0]
#     #         images.append((img_path, label))
        
#     #     return images

#     # def _split_train_test(self, all_images):
#     #     split_file = os.path.join(self.root_dir, 'train_test_split.txt')
#     #     split_df = pd.read_csv(split_file, sep=' ', header=None, names=['id', 'is_train'])
#     #     train_images = []
#     #     test_images = []
        
#     #     for img, label in all_images:
#     #         img_id = int(re.findall(r'\d+', os.path.basename(img))[0] )
#     #         is_train = split_df[split_df['id'] == img_id]['is_train'].values[0]          
#     #         if is_train == 1:
#     #             train_images.append((img, label))
#     #         else:
#     #             test_images.append((img, label))
        
#     #     return train_images, test_images

#     def train_dataloader(self):

#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

#     # def val_dataloader(self):
#     #     return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)




if __name__ == '__main__':
    dm = CubFewShotDataModule('/home/WRS/ProtosViT/datasets/cub')
    dm.setup()
    for data in dm.train_dataloader():
        print(data[0].shape)
        print(data[1])
        support_labels = data[1][:25]
        query_labels = data[1][25:]
        positive_mask = query_labels.unsqueeze(1) == support_labels.unsqueeze(0)
        print(positive_mask.shape)