import os
import h5py
import random
import numpy as np
import sys

from collections import Counter

from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
from torchtext.vocab import Vocab

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from src.models.lit_baseline import LitBaseline

def load_flatvec_data(data_dir):
    temp_data = pkl.load(open(data_dir,"rb"))
    np_data = [np.array(x) for x in temp_data]
    data = np.vstack(np_data)
    note_vecs, chords = data[:, :-1], data[:,-1]
    note_vecs = np.asarray(note_vecs, dtype = np.float32)

    return note_vecs, chords


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

    data_root = conf.dataset.directory
    dataset_name = conf.dataset.name

    if data_conf.subset == "full":
        pass

    elif data_conf.subset == "toy":
        pass

    # Do we need to get vocab for train/val/test individually?
    
    note_vec, chords = load_flatvec_data(data_root)
    chord_vocab = Vocab(Counter(chords))
    encoded_chords = [chord_vocab.stoi[ch] for ch in chords]

    model = LitBaseline(conf)

    # train_loader = DataLoader(tasksets.train, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)
    # val_loader = DataLoader(tasksets.validation, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)
    # test_loader = DataLoader(tasksets.test, batch_size = data_conf.meta_batch_size, shuffle = data_conf.shuffle_train,  num_workers = data_conf.num_workers)


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
    