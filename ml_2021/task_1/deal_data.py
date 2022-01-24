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
        pd_data = pd.read_csv("./covid.train.csv",index_col=["id"])
        data_y = pd_data.drop("")

        with open("./covid.train.csv", 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)


        # Training data (train/dev sets)
        # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
        target = data[:, -1]
        data = data[:, list(range(93))]
        print(data)
        print(target)

        # Splitting training data into train & dev sets
        if mode == 'train':
            indices = [i for i in range(len(data)) if i % 10 != 0]
        elif mode == 'dev':
            indices = [i for i in range(len(data)) if i % 10 == 0]

        # Convert data into PyTorch tensors
        self.data = torch.FloatTensor(data[indices])
        self.target = torch.FloatTensor(target[indices])
        print(self.data)
        print(self.target)
        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format("train", len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def get_dataloader():
    dataset = COVID19Dataset()
    dataloader = DataLoader(dataset, batch_size=256,shuffle=True)
    return dataloader

if __name__ == '__main__':
    tr_set = get_dataloader()
