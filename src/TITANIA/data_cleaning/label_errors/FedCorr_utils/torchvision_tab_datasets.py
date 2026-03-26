import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, values_df, targets_df,):
        vals = values_df.values.astype('float32')
        self.data = torch.tensor(vals, dtype=torch.float32)
        t_list = list(targets_df.values.astype('int32'))
        self.targets = [j for sub in t_list for j in sub]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
