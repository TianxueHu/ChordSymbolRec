import os
import h5py
import random
import numpy as np
import sys

from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.models.lit_baseline import LitMAML

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)

if __name__ == "__main__":
    
    config_path = sys.argv[1]
    conf = OmegaConf.load(config_path)
    data_conf = conf.dataset

    torch.manual_seed(conf.experiment.seed)
    np.random.seed(conf.experiment.seed)
    random.seed(conf.experiment.seed)

    if conf.experiment.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    ways = conf.dataset.ways
    shots = conf.dataset.shots
    data_root = conf.dataset.directory
    dataset_name = conf.dataset.name

    if dataset_name == "mini-imagenet":
        tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2*shots,
                                                  train_ways=ways,
                                                  test_samples=2*shots,
                                                  test_ways=ways,
                                                  num_tasks=conf.dataset.task_num,
                                                  root=data_root)
    elif dataset_name == "FC100":
        raise NotImplementedError

    else:
        raise NotImplementedError

    if data_conf.subset == "full":
        pass

    elif data_conf.subset == "toy":
        pass

    model = LitMAML(conf)
    train_loader = DataLoader(tasksets.train, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)
    val_loader = DataLoader(tasksets.validation, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)
    test_loader = DataLoader(tasksets.test, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)


    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(conf.logging.output_dir,conf.experiment.objective))
    trainer = pl.Trainer(
        gpus= conf.experiment.gpu_num,
	    # accelerator='ddp',
        # replace_sampler_ddp=replace_sampler_ddp,
        max_epochs=conf.training.epochs,
        progress_bar_refresh_rate=conf.logging.progress_bar_refresh_rate, 
        logger=tb_logger, 
        val_check_interval=conf.validation.check_interval, 
        # limit_val_batches=conf.validation.check_ratio # remember to shuffle val to enable val with different subset
    )

    # Train the model ⚡
    trainer.fit(model, train_loader, val_loader)
    