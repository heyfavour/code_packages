import time, datetime
import torch
import torch.nn as nn
import numpy as np

from deal_data import train_data_loader, validation_data_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
        self.convolution = nn.Sequential(
            # 卷积1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # output [64 128 128]
            nn.BatchNorm2d(64),  # 归一化 [64 256 256]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [64 64 64]
            # 卷积2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # output [128 64 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [128 32 32]
            # 卷积3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # output [256 32 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [256 16 16]
            # 卷积4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # output [512 16 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [512 8 8]
            # 卷积5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # output [512 8 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [512 4 4]
        )
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.sigmod(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        conv = self.convolution(x)
        flat = conv.view(conv.size()[0], -1)
        #flat = self.flatten(conv)
        label = self.cnn(flat)
        return label




if __name__ == '__main__':
    cuda = True

    train_loader, train_num = train_data_loader()
    validation_loader, validation_num = validation_data_loader()

    cnn = CNN()
    if torch.cuda.is_available() and cuda: cnn = cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    loss_calc = nn.CrossEntropyLoss()

    epoch_size = 32
    for epoch in range(epoch_size):
        print(f"[{epoch + 1:0=3d}/{epoch_size}] epoch train start")
        cnn.train()
        start = datetime.datetime.now()
        for step, (train_x, train_y) in enumerate(train_loader):
            optimizer.zero_grad()
            if cuda:
                Y = cnn(train_x.cuda())
                loss = loss_calc(Y, train_y.cuda())
            else:
                Y = cnn(train_x)
                loss = loss_calc(Y, train_y)
            loss.backward()
            optimizer.step()

        Y_right_sum = 0
        Y_loss_sum = 0.0
        end = datetime.datetime.now()
        print(f"[{epoch:0=3d}/{epoch_size}] epoch train end,using time {end - start}")

        cnn.eval()
        with torch.no_grad():
            for index, (validation_x, validation_y) in enumerate(validation_loader):
                if cuda:
                    Y = cnn(validation_x.cuda())
                    validation_loss = loss_calc(Y, validation_y.cuda())
                    Y = Y.cpu()
                else:
                    Y = cnn(validation_x)
                    validation_loss = loss_calc(Y, validation_y)
                Y_right_sum = Y_right_sum + np.sum(np.argmax(Y.data.numpy(), axis=1) == validation_y.numpy())
                Y_loss_sum = Y_loss_sum + validation_loss.item()
            info = f"[{(epoch + 1):0=3d}/{32:0=3d}]  ACC: {Y_right_sum * 100 / train_num:3.2f}% LOSS: {Y_loss_sum / validation_num:3.6f}"
            print(info)
