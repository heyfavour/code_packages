import os
import csv
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''

    def __init__(self):
        mode = "train"
        pd_data = pd.read_csv("./covid.train.csv", index_col=["id"])
        data_y = pd_data["tested_positive.2"]
        pd_data.drop("tested_positive.2", axis=1, inplace=True)
        self.data = torch.FloatTensor(pd_data.to_numpy())
        self.target = torch.FloatTensor(data_y.to_numpy())
        # 归一化
        mean = self.data[:, 40:].mean(dim=0, keepdim=True)
        std = self.data[:, 40:].std(dim=0, keepdim=True)
        self.data[:, 40:] = (self.data[:, 40:] - mean) / std

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def get_dataloader():
    dataset = COVID19Dataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()
