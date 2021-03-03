import os
import h5py
import numpy as np
import sys

from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from av_sync.models.lit_baseline import LitBaseline
from av_sync.datasets.torch_datasets import AVOTF_Dataset

import pickle as pkl


if __name__ == "__main__":
    
    hparams_path, checkpoint_path = sys.argv[1], sys.argv[2]
    all_conf = OmegaConf.load(hparams_path)
    conf = all_conf.configs
    # print(conf)
    data_conf = conf.dataset

    torch.manual_seed(conf.experiment.seed)
    np.random.seed(conf.experiment.seed)
    if conf.experiment.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if data_conf.fake_label == "binary":
        return_binary = True
    else:
        return_binary = False

    chunk_size = data_conf.chunk_size
    fps = data_conf.fps
    sample_rate = data_conf.sample_rate
    shift_low = data_conf.shift_low
    shift_high = data_conf.shift_high

    silero_model, _, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                            model='silero_stt',
                            language='en',
                            device = device)
    silero_model.eval()

    train_dataset = AVOTF_Dataset(os.path.join(data_conf.directory, "train"), chunk_size, fps, sample_rate, device, silero_model, shift_low_bound = shift_low, shift_up_bound = shift_high, return_binary = return_binary, random_seed = conf.experiment.seed)
    val_dataset = AVOTF_Dataset(os.path.join(data_conf.directory, "val"), chunk_size, fps, sample_rate, device, silero_model, shift_low_bound = shift_low, shift_up_bound = shift_high, return_binary = return_binary, eval_mode = True, random_seed = conf.experiment.seed)
    
    if data_conf.subset == "full":
        train_loader = DataLoader(train_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_train, num_workers = data_conf.num_workers)
        val_loader = DataLoader(val_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_val, num_workers = data_conf.num_workers) 
        replace_sampler_ddp = False

    elif data_conf.subset == "toy":
        with h5py.File(os.path.join(data_conf.directory, data_conf.real_train_fname), "r") as real_train_f, h5py.File(os.path.join(data_conf.directory, data_conf.real_val_fname), "r") as real_val_f: 

            fake_train_index = real_train_f["video"].shape[0]
            fake_val_index = real_val_f["video"].shape[0]

        
        toy_train_index = np.hstack((np.arange(20), np.arange(fake_train_index,fake_train_index + 20)))
        toy_val_index = np.hstack((np.arange(20), np.arange(fake_val_index,fake_val_index + 20)))
        train_loader = DataLoader(train_dataset, batch_size = data_conf.batch_size, sampler = SubsetRandomSampler(toy_train_index), num_workers = data_conf.num_workers)
        val_loader = DataLoader(val_dataset, batch_size = data_conf.batch_size, sampler = SubsetRandomSampler(toy_val_index), num_workers = data_conf.num_workers)
        replace_sampler_ddp = True

    video_shape = train_dataset[0][0].shape
    audio_shape = train_dataset[0][1].shape

    model = LitBaseline.load_from_checkpoint(checkpoint_path, hparams_file= hparams_path)

    # model = LitBaseline(all_conf.video_size, all_conf.audio_size, conf)
    trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path)
    trainer.test(model, test_dataloaders = val_loader)

    # tb_logger = pl_loggers.TensorBoardLogger(os.path.join(conf.logging.output_dir,conf.experiment.objective))

    pkl.dump([model.test_predictions, model.test_shifts, model.test_labels], open("results.pkl", "wb"))
