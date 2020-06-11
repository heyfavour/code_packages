import torch
import pandas as pd
import numpy as np
from deal_data import  get_pd_csv,get_csv_data

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=1,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


def basic_data_prepare(code):
    get_pd_csv(code)
    get_csv_data(code)

def series_to_supervised(df):
    x_np = []
    y_np = []
    for index,i in enumerate(df[:-35,:]):
        x_np.append(np.concatenate(([i], df[index+1:index+30]), ))
        y_np.append([df[index+30][3],df[index+32][3],df[index+34][3]])
    x_np = np.array(x_np)
    y_np = np.array(y_np)
    return x_np,y_np


def get_train_data():
    df = pd.read_csv("train_data.csv").set_index('date').to_numpy()
    x_np,y_np = series_to_supervised(df)
    return x_np,y_np


if __name__ == '__main__':
    #code = "sz.002223"
    #basic_data_prepare(code)
    x_np,y_np = get_train_data()
