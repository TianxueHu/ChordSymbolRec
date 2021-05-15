from collections import Counter
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

from torchtext.vocab import Vocab

from chord_rec.models.lit_baseline import LitBaseline
from chord_rec.datasets.vec_datasets import FFNNDataset

import random
import os
import sys
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class CheckpointEveryNEpoch(pl.Callback):
    def __init__(self, start_epoc, ckpt_every_n = 1):
        self.start_epoc = start_epoc
        self.ckpt_every_n = ckpt_every_n

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        # file_path = f"{trainer.logger.log_dir}/checkpoints/epoch={trainer.current_epoch}.pt"
        epoch = trainer.current_epoch
        if epoch >= self.start_epoc and epoch % self.ckpt_every_n == 0:
            ckpt_path = f"{trainer.logger.log_dir}/checkpoints/epoch={epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)
            
early_stop_callback = EarlyStopping(
   monitor='val_acc',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='max'
)


if __name__ == "__main__":
    

    ckp_dir = sys.argv[1]
    hparams_path = os.path.join(ckp_dir, "hparams.yaml")
    checkpoint_path = os.path.join(ckp_dir, "checkpoints", sys.argv[2])

    all_conf = OmegaConf.load(hparams_path)
    conf = all_conf.configs
    data_conf = conf.dataset

    seed = conf.experiment.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if conf.experiment.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_root = conf.dataset.directory
    dataset_name = conf.dataset.name

    train_path = os.path.join(data_root, conf.dataset.train_fname)
    val_path = os.path.join(data_root, conf.dataset.val_fname)
    test_path = os.path.join(data_root, conf.dataset.test_fname)

    train = pkl.load(open(train_path,"rb"))
    train = [np.array(x) for x in train]
    train = np.vstack(train)
    train_notes, train_chords = train[:, :-1], train[:,-1]

    val = pkl.load(open(val_path,"rb"))
    val = [np.array(x) for x in val]
    val = np.vstack(val)
    val_notes, val_chords = val[:, :-1], val[:,-1]

    test = pkl.load(open(test_path,"rb"))
    test = [np.array(x) for x in test]
    test = np.vstack(test)
    test_notes, test_chords = test[:, :-1], test[:,-1]


    chords = np.hstack([train_chords, val_chords, test_chords])
    chord_vocab = Vocab(Counter(chords))

    # encoded_train_chords = [chord_vocab.stoi[ch] for ch in train_chords]
    # encoded_val_chords = [chord_vocab.stoi[ch] for ch in val_chords]
    encoded_test_chords = [chord_vocab.stoi[ch] for ch in test_chords]

    # train_dataset = FFNNDataset(train_notes, encoded_train_chords)
    # val_dataset = FFNNDataset(val_notes, encoded_val_chords)
    test_dataset = FFNNDataset(test_notes, encoded_test_chords)

    vec_size = len(train_notes[0])
    vocab_size = vocab_size = len(chord_vocab.stoi)

    # train_loader = DataLoader(train_dataset, batch_size =data_conf.batch_size, shuffle = data_conf.shuffle_train, num_workers = data_conf.num_workers, drop_last = True)
    # val_loader = DataLoader(val_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_val, num_workers = data_conf.num_workers, drop_last = True)
    test_loader =  DataLoader(test_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_val, num_workers = data_conf.num_workers, drop_last = True)

    model = LitBaseline.load_from_checkpoint(checkpoint_path, chord_vocab = chord_vocab)

    trainer = pl.Trainer()
    trainer.test(model, test_dataloaders = test_loader)