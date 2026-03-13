import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import hydra
import pickle
import argparse
import numpy as np
import pyrootutils
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from os.path import join as pj
from hydra import compose, initialize
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from models.ClassificationModulePrune import ClassificationModulePrototype
from data.CubDataset_fsl import MetaLearningDataLoader

pyrootutils.setup_root(os.getcwd(), indicator="requirements.txt", pythonpath=True)

def load_model_dataset(path_sim: str, set: str, batch_size: Optional[int] = None, num_workers: Optional[int] = None, cfg: Optional[DictConfig] = None, **kwargs,):
    """
    Load model and dataset from a given path.
    Args:
        path_sim (str): Path to the simulation
        set (str): Set to load (train, val, test).
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        **kwargs: Additional arguments.
    Returns:
        model (LightningModule): Model.
        dataset (LightningDataModule): Dataset.
    """
    kwargs_datamodule = {}
    if batch_size is not None:
        kwargs_datamodule["batch_size"] = batch_size
    if num_workers is not None:
        kwargs_datamodule["num_workers"] = num_workers
    config_path = os.path.relpath(pj(path_sim, ".hydra"), os.path.dirname(os.path.abspath(__file__)))
    if cfg is None:
        with initialize(version_base=None, config_path=config_path):
            cfg = compose(config_name="config.yaml")
    #datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, *kwargs_datamodule)
    
    datamodule = MetaLearningDataLoader(data_dir='/home/WRS/ProtosViT/datasets/cub/CUB_200_2011',n_way=5,k_shot=5,q_query=15,n_iter=500)

    # this is a bit of hack as loading the model without hydra fails due to import
    model: LightningModule = hydra.utils.instantiate(cfg.model, img_size=datamodule.dims, num_classes=datamodule.num_classes)
    model_checkpoint = [x for x in os.listdir(os.path.abspath(pj(path_sim, "checkpoints"))) if (x.endswith(".ckpt") & ("last" not in x))][-1]
    model = ClassificationModulePrototype.load_from_checkpoint(pj(path_sim, "checkpoints", model_checkpoint))
    model.eval()
    model.freeze()
    if set == "train":
        datamodule.setup(stage="fit")
        dataset = datamodule.train_dataloader()
    elif set == "val":
        datamodule.setup(stage="fit")
        dataset = datamodule.val_dataloader()
    elif set == "test":
        datamodule.setup(stage="test")
        dataset = datamodule.test_dataloader()
    return model, dataset

def save_results(path_sim: str | Path, set: str, cfg: Optional[DictConfig] = None) -> bool:
    model, dataloader = load_model_dataset(path_sim, set, cfg=cfg)
    labels = []
    preds = []
    importance = []
    similarity_prototype = []
    similarity_background = []
    samples = []
    for i, (sample, label,attr) in tqdm(enumerate(dataloader), total=len(dataloader)):
        sample = sample.cuda()
        with torch.no_grad():
            tmp = model.cuda()(sample)
        samples.append(sample.cpu().numpy())
        preds.append(tmp["pred"].argmax(dim=1).detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        importance.append(tmp["importance"].detach().cpu().numpy())
        similarity_prototype.append(tmp["similarity_prototype"].detach().cpu().numpy())
        similarity_background.append(tmp["similarity_background"].detach().cpu().numpy())
        del sample
        del tmp
    dict_results: dict(str, np.ndarray) = {
        "sample": np.concatenate(samples),
        "labels": np.concatenate(labels),
        "preds": np.concatenate(preds),
        "importance": np.concatenate(importance),
        "similarity_prototype": np.concatenate(similarity_prototype),
        "similarity_background": np.concatenate(similarity_background),
    }
    if not isinstance(path_sim, Path):
        path_sim = Path(path_sim)
    pkl_path = path_sim / f"results_{set}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(dict_results, f)
    return True

def compute_model_stats(path_sim: str | Path, threshold: float = 0.0):
    if not isinstance(path_sim, Path):
        path_sim = Path(path_sim)
    pkl_path = list(path_sim.glob("*.pkl"))
    if len(pkl_path) == 0:
        msg = f"no pkl file found in {path_sim}"
        raise FileNotFoundError(msg)
    elif len(pkl_path) > 1:
        msg = f"more than one pkl file found in {path_sim}"
        raise FileNotFoundError(msg)
    pkl_path = pkl_path[0]
    with pkl_path.open("rb") as f:
        dict_results = pickle.load(f)
        labels = dict_results["labels"]
    importance = dict_results["importance"].copy()
    if threshold > 0:
        importance[importance < threshold] = 0
        preds = importance.sum(axis=1).argmax(axis=1)
    else:
        preds = dict_results["preds"]
    accuracy = (preds == labels).sum() / len(labels)
    class_importance = importance[np.arange(importance.shape[0]), :, labels]
    local_size = (class_importance > 0).sum(axis=1).mean()
    global_size = ((class_importance > 0).sum(axis=0) > 0).sum()
    # balanced_accuracy = balanced_accuracy_score(labels, preds)
    return accuracy, local_size, global_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_sim", type=str, help="Path to the simulation")
    args = parser.parse_args()
    print(f"Computing results for test set of {args.path_sim}")
    save_results(args.path_sim, "test")