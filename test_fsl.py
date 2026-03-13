import torch
import hydra
import numpy as np
import pyrootutils
import shared_utils
import pytorch_lightning as pl
from omegaconf import DictConfig
from typing import List, Optional, Tuple
from pytorch_lightning.loggers import Logger
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
log = shared_utils.get_pylogger(__name__)

from data.CubDataset_fsl import MetaLearningDataLoader
from models.ClassificationModulePrune_FSL import ClassificationModulePrototype_FSL
from pytorch_lightning.callbacks import ModelCheckpoint
from data.Chestx_fsl import ChestX_DataLoader
from data.Cifarfs import Cifarfs_DataLoader
from data.EuroSAT_FSL import EuroSAT_DataLoader 
from data.PlantVillage_fsl import PlantVillage_DataLoader   
from data.stanfordcars_fsl import stanfordcars_DataLoader
from data.aircraft_fsl import aircraft_DataLoader
from data.miniimagenet_fsl import miniimagenet_DataLoader


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.autograd.set_detect_anomaly(True)
os.environ["HYDRA_FULL_ERROR"] = "1"

class Datum:
    def __init__(self, impath, label, classname):
        self.impath = impath      # 图像路径
        self.label = label        # 类别标签
        self.classname = classname 

def train_fsl(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    Trains the model in few-shot learning (FSL) setup. Can additionally evaluate on a testset,
    using best weights obtained during training.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    seed = None
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        np.random.seed(cfg.seed)
        seed = cfg.seed

    # Instantiate the FSL datamodule (using a sampler such as CategoriesSampler)
    log.info(f"Instantiating FSL datamodule <{cfg.data._target_}>")
    log.info(f"Instantiating FSL datamodule <CUBFewShotDataModule>")
    datamodule = MetaLearningDataLoader(data_dir='/home/WRS/ProtosViT/datasets/cub/CUB_200_2011',n_way=10,k_shot=5,q_query= 15 ,n_iter=1)
    #datamodule_ft = MetaLearningDataLoader_ft(data_dir='/home/WRS/ProtosViT/datasets/cub/CUB_200_2011',n_way=5,k_shot=5,q_query= 15 ,n_iter=500)
    #datamodule = Cifarfs_DataLoader(data_dir='/home/WRS/ProtosViT/datasets/cifar-fs/CIFAR-FS/cifar100',transform=None,n_way=5, k_shot=20, q_query= 15 ,n_iter=500)
    #datamodule = miniimagenet_DataLoader(data_dir='/home/WRS/ProtosViT/datasets/miniImageNet',transform=None,n_way=5, k_shot=20, q_query= 15 ,n_iter=500)
    log.info(f"Instantiating FSL model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        img_size=datamodule.dims,
        num_classes=datamodule.num_classes,  # Usually n_way in FSL
        weight_class=datamodule.weight_class,
        data_mean=datamodule.mean,
        data_std=datamodule.std,
    )


    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = shared_utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = shared_utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, num_sanity_val_steps=0)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        shared_utils.log_hyperparameters(object_dict)

    # Compile the model if specified
    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    # Training phase for FSL
    if cfg.get("train"):
        log.info("Starting FSL training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),  # Continue from a checkpoint if specified
        )
    
    train_metrics = trainer.callback_metrics

    # Testing phase for FSL
    if cfg.get("test"):
        log.info("Starting FSL testing!")
        #ckpt_path = trainer.checkpoint_callback.best_model_path
        ckpt_path = "logs/train/neurips/dino_loramoeffn_ags_1.4multilevel_3e-3infonce_1shot_83.43/checkpoints/epoch_079.ckpt"
        if ckpt_path == "":
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None

        # model = torch.load(ckpt_path, map_location='cpu')
        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #     model = model.module
        # msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
        # log.info(msg)
        #trainer.test(model=model, datamodule=datamodule, ckpt_path=None) 
        # trainer.fit(model=model_ft,datamodule=datamodule,ckpt_path=ckpt_path) # Continue from a checkpoint if specified
        # ckpt_path = trainer.checkpoint_callback.best_model_path

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best checkpoint path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    # Save results
    shared_utils.save_results(cfg.paths.output_dir, "test", cfg)

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="configs", config_name="test_fsl.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    shared_utils.extras(cfg)
    train_fsl(cfg)

if __name__ == "__main__":
    main()
