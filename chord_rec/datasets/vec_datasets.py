from torch.utils.data import Dataset

class Vec45Dataset(Dataset):
    def __init__(self, note_vec_seq, chord_seq, vocab):
        'Initialization'
        self.note_vec_seq = note_vec_seq
        self.chord_seq = chord_seq
        self.vocab = vocab

    def __len__(self):
        'Get the total length of the dataset'
        return len(self.note_vec_seq)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        
        return self.note_vec_seq[index], self.vec_encode(self.chord_seq[index])
    
    def encode(self, x):
        return self.vocab.stoi[x]


    def vec_encode(self, x):
        return np.vectorize(self.encode)(x)
    

    def decode(self, x):
        return self.vocab.itos[x]


    def vec_decode(self, x):
        return np.vectorize(self.decode)(x)