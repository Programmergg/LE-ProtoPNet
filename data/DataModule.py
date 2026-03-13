import torch
import numpy as np
from typing import Optional
import pytorch_lightning as pl
from data import create_data_set
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_dir: str = "./data", download=False, img_size=224, batch_size: int = 64, seed: Optional[int] = None, valid_split: Optional[float] = None,
                 weighted_sampler: Optional[bool] = False, photometrics_augmentation: Optional[bool] = False, augmentation_severity: Optional[int] = 6):
        super().__init__()
        self.seed = seed
        self.weight_class = None
        self.data_dir = data_dir
        self.download = download
        self.img_size = img_size
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.dims = (3, img_size, img_size)
        self.weighted_sampler = weighted_sampler
        self.augmentation_severity = augmentation_severity
        self.photometrics_augmentation = photometrics_augmentation

        dataset = create_data_set(dataset)
        self.std = dataset.std
        self.mean = dataset.mean
        self.Dataset = dataset.dataset_class
        self.num_classes = dataset.num_classes
        self.use_keyword_split = dataset.use_keyword_split
        if download:
            self.Dataset(root=data_dir, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
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
        if self.photometrics_augmentation:
            photometric_transforms = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    # transforms.RandomGrayscale(0.1),
                    transforms.RandomApply([transforms.GaussianBlur((5, 5))]),
                ]
            )
        else:
            photometric_transforms = None
        if stage == "fit" or stage is None:
            if self.use_keyword_split:
                train_dataset = self.Dataset(
                    root=self.data_dir,
                    split="train",
                    transform=train_transform,
                    photometric_transforms=photometric_transforms,
                )
                if self.valid_split is not None:
                    val_dataset = self.Dataset(
                        root=self.data_dir,
                        split="train",
                        transform=test_transform,
                        photometric_transforms=photometric_transforms,
                    )
            else:
                train_dataset = self.Dataset(
                    root=self.data_dir,
                    train=True,
                    transform=train_transform,
                    photometric_transforms=photometric_transforms,
                )
                if self.valid_split is not None:
                    val_dataset = self.Dataset(
                        root=self.data_dir,
                        train=True,
                        transform=test_transform,
                        photometric_transforms=photometric_transforms,
                    )
            if self.valid_split is not None:
                if self.seed is not None:
                    pl.seed_everything(self.seed)
                self.train_set, _ = random_split(train_dataset, [int(len(train_dataset) * (1 - self.valid_split)), len(train_dataset) - int(len(train_dataset) * (1 - self.valid_split))])
                if self.seed is not None:
                    pl.seed_everything(self.seed)
                _, self.val_set = torch.utils.data.random_split(val_dataset, [int(len(train_dataset) * (1 - self.valid_split)), len(train_dataset) - int(len(train_dataset) * (1 - self.valid_split))])
            else:
                self.train_set = train_dataset
            if self.weighted_sampler:
                self.weight_class = get_weight_class(self.train_set)
                self.train_sampler = torch.utils.data.WeightedRandomSampler(weights=self.weight_class, num_samples=len(self.train_set), replacement=True)
            else:
                self.train_sampler = None

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if self.use_keyword_split:
                self.test_set = self.Dataset(root=self.data_dir, split="test", transform=test_transform)
            else:
                self.test_set = self.Dataset(root=self.data_dir, train=False, transform=test_transform)

    def train_dataloader(self):
        shuffle_train = self.train_sampler is None
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle_train, num_workers=6, sampler=self.train_sampler)

    def val_dataloader(self):
        if self.valid_split is not None:
            return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return []

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def unnormalize(self, batch):
        unnormalize = transforms.Normalize((-self.mean / self.std), (1.0 / self.std))
        return unnormalize(batch)

def get_weight_class(dataset):
    print("Getting weight class for random weighted sampler")
    target = [dataset[i][1] for i in range(len(dataset))]
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1.0 / class_sample_count
    samples_weight = weight[target]
    return torch.from_numpy(samples_weight)