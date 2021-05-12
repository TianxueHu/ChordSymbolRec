import os
import h5py
import random
import numpy as np
import sys
import pickle as pkl

from collections import Counter

from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, random_split
from torchtext.vocab import Vocab

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sklearn.model_selection import train_test_split

from chord_rec.models.lit_seq2seq import LitSeq2Seq

from chord_rec.datasets.vec_datasets import Vec45Dataset

def load_flatvec_data(data_dir):
    temp_data = pkl.load(open(data_dir,"rb"))
    np_data = [np.array(x) for x in temp_data]
    data = np.vstack(np_data)
    note_vecs, chords = data[:, :-1], data[:,-1]
    note_vecs = np.asarray(note_vecs, dtype = np.float32)

    return note_vecs, chords

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


if __name__ == "__main__":
    
    ckp_dir = sys.argv[1]
    ckp_version = sys.argv[2]
    hparams_path = os.path.join(ckp_dir, "hparams.yaml")
    checkpoint_path = os.path.join(ckp_dir, "checkpoints", ckp_version)

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

    if data_conf.subset == "full":
        pass

    elif data_conf.subset == "toy":
        pass
    
    ### Prepare the data ###

    # Do we need to get vocab for train/val/test individually?
    if conf.model.type == "ffnn":
        note_vec, chords = load_flatvec_data(data_root)
        chord_vocab = Vocab(Counter(chords))
        encoded_chords = [chord_vocab.stoi[ch] for ch in chords]
  
        model = LitBaseline(conf)

        raise NotImplementedError

    elif conf.model.type in ["s2s", "attn_s2s", "transformer"]:

        data = pkl.load(open(data_conf.fpath, "rb"))

        note_seq, chord_seq = [],[]
        max_seq_len = 0
        data_num = 0
        for file in data:
            data_num += len(file)
            for window in file:
                note_seq.append(window[0])
                chord_seq.append(window[1])
                max_seq_len = max(max_seq_len, len(window[1]))

        note_padding_vec = np.full(len(note_seq[0][0]), -1).reshape(1,-1) # should be 45; not sure if -1 is good
        note_ending_vec = np.ones(len(note_seq[0][0])).reshape(1,-1) # should be 45
        note_starting_vec = np.zeros(len(note_seq[0][0])).reshape(1,-1) # should be 45

        chord_start = "<sos>"
        chord_padding = "<pad>"
        chord_end = "<eos>"

        padded_note_seq = []
        padded_chord_seq = []

        eval_masks = []

        for i in range(len(note_seq)):
            len_diff = max_seq_len - len(note_seq[i])

            temp_note_vec = np.vstack((note_starting_vec, np.array(note_seq[i]), note_ending_vec, np.repeat(note_padding_vec, len_diff , axis = 0)))
            padded_note_seq.append(temp_note_vec)
            
            eval_masks.append([False] + [True for _ in range(len(note_seq[i]))] + [False for _ in range(len_diff+1)])
            temp_chord_vec = np.hstack((chord_start, np.array(chord_seq[i]), chord_end, np.repeat(chord_padding, len_diff , axis = 0)))
            padded_chord_seq.append(temp_chord_vec)
        
        eval_masks = np.array(eval_masks)
        stacked_note_seq = np.stack(padded_note_seq, axis = 0)
        stacked_chord_seq = np.vstack(padded_chord_seq)

        note_vec = np.asarray(stacked_note_seq, dtype = np.float32)
        chord_vocab = Vocab(Counter(list(stacked_chord_seq.flatten())))

        vec_size = note_vec.shape[-1]
        vocab_size = len(chord_vocab.stoi)

        assert data_conf.val_ratio + data_conf.test_ratio <= 0.6, "At least 40 percent of the data needed for training"

        dataset = Vec45Dataset(note_vec, stacked_chord_seq, eval_masks, chord_vocab)

        
        train_ratio = 1 - data_conf.val_ratio - data_conf.test_ratio

        train_len = int(len(dataset)*train_ratio)
        val_len = int(len(dataset)*data_conf.val_ratio)
        test_len = len(dataset) - train_len - val_len

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len], 
                                                        generator=torch.Generator().manual_seed(seed)
                                                       )


        train_loader = DataLoader(train_dataset, batch_size =data_conf.batch_size, shuffle = data_conf.shuffle_train, num_workers = data_conf.num_workers, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_val, num_workers = data_conf.num_workers, drop_last = True)
        test_loader =  DataLoader(test_dataset, batch_size = data_conf.batch_size, shuffle = data_conf.shuffle_val, num_workers = data_conf.num_workers, drop_last = True)

        MAX_LEN = max_seq_len + 2

        if conf.model.type == "attn_s2s":
            model = LitSeq2Seq(vec_size, MAX_LEN, chord_vocab, conf)
        else:
            raise NotImplementedError

    epochs = conf.training.warm_up + conf.training.decay_run + conf.training.post_run

    tb_logger = pl_loggers.TensorBoardLogger(conf.logging.output_dir, name = conf.experiment.objective)
    trainer = pl.Trainer(
        resume_from_checkpoint = checkpoint_path,
        gpus= conf.experiment.gpu_num,
	    # accelerator='ddp',
        # replace_sampler_ddp=replace_sampler_ddp,
        max_epochs = epochs,
        progress_bar_refresh_rate=conf.logging.progress_bar_refresh_rate, 
        logger=tb_logger, 
        val_check_interval=conf.validation.check_interval, 
        gradient_clip_val= 0.5,
        stochastic_weight_avg=True,
        callbacks=[CheckpointEveryNEpoch(0, conf.training.save_every_n)]
        # limit_val_batches=conf.validation.check_ratio # remember to shuffle val to enable val with different subset
    )

    # Train the model ⚡
    trainer.fit(model, train_loader, val_loader)
    