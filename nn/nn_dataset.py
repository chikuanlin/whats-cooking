import torch
from torch.utils.data import Dataset

class NNDataset(Dataset):

    def __init__(self, x, y):
        self.data = [(x[i, :], y[i]) for i in range(y.shape[0])]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index][0]), torch.tensor(self.data[index][1], dtype=torch.long)

    def __len__(self):
        return len(self.data)
