from dataclasses import dataclass
from typing import Callable, Optional
from torchvision.datasets import Flowers102

from .CubDataset import CubDataset
from .IsicDataset import IsicDataset
from .PetDataset import OxfordIIITPet
from .LungColonDataset import LungColonDataset
from .FunnybirdsDataset import FunnyBirdsDataset
from .StanfordCarsDataset import StanfordCarsDataset
from .RSNAPneumoniaDataset import RSNAPneumoniaDataset

@dataclass
class DataConfig:
    dataset_class: Callable
    num_classes: int
    mean: list[float]
    std: list[float]
    use_keyword_split: Optional[bool] = False

dataset_index: dict[str, DataConfig] = {
    "cub": DataConfig(
        CubDataset,
        num_classes=200,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    "stanford_cars": DataConfig(
        StanfordCarsDataset,
        num_classes=196,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_keyword_split=True,
    ),
    "flowers102": DataConfig(
        Flowers102,
        num_classes=102,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_keyword_split=True,
    ),
    "OxfordPet": DataConfig(
        OxfordIIITPet,
        num_classes=37,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_keyword_split=True,
    ),
    "funny_birds": DataConfig(
        FunnyBirdsDataset,
        num_classes=50,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        use_keyword_split=True,
    ),
    "rsna": DataConfig(
        RSNAPneumoniaDataset,
        num_classes=2,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
    "isic2019": DataConfig(
        IsicDataset,
        num_classes=9,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
    "lung_image": DataConfig(
        LungColonDataset,
        num_classes=3,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        use_keyword_split=True,
    ),
    "colon_image": DataConfig(
        LungColonDataset,
        num_classes=2,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        use_keyword_split=True,
    )
}

def create_data_set(dataset_name: str):
    if dataset_name in dataset_index.keys():
        return dataset_index[dataset_name]
    else:
        raise ValueError("Invalid dataset name")