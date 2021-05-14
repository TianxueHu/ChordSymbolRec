from torch.utils.data import Dataset
import numpy as np
class Vec45Dataset(Dataset):
    def __init__(self, note_vec_seq, chord_seq, eval_mask, vocab):
        'Initialization'
        self.note_vec_seq = note_vec_seq
        self.chord_seq = chord_seq
        self.vocab = vocab
        self.eval_mask = eval_mask

    def __len__(self):
        'Get the total length of the dataset'
        return len(self.note_vec_seq)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        
        return self.note_vec_seq[index], self.vec_encode(self.chord_seq[index]), self.eval_mask[index]
    
    def encode(self, x):
        return self.vocab.stoi[x]


    def vec_encode(self, x):
        return np.vectorize(self.encode)(x)
    

    def decode(self, x):
        return self.vocab.itos[x]


    def vec_decode(self, x):
        return np.vectorize(self.decode)(x)


class FFNNDataset(Dataset):
    def __init__(self, note_vec, chords):
        'Initialization'
        self.note_vec = note_vec
        self.chords = chords

    def __len__(self):
        'Get the total length of the dataset'
        return len(self.note_vec)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.note_vec[index].astype(np.float32), self.chords[index]