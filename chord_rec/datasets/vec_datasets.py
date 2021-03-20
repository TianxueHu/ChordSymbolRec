from torch.utils.data import Dataset

class FlatVecDataset(Dataset):
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
        return self.note_vec[index], self.chords[index]