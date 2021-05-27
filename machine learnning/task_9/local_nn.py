import random
import torch
import torch.nn as nn
import numpy as np

from deal_data import deal_data
from torch.utils.data import DataLoader


class AutoEencoder(nn.Module):
    def __init__(self):
        super(AutoEencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x


def same_seeds(seed):
    # 每次运行网络的时候相同输入的输出是固定的
    torch.manual_seed(seed)  # 初始化种子保持一致
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.初始化种子保持一致
    random.seed(seed)  # Python random module. 初始化种子保持一致
    # 内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    torch.backends.cudnn.benchmark = False
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    same_seeds(0)

    model = AutoEencoder().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    model.train()
    img_dataloader = deal_data()

    epoch_loss = 0

    # 主要的訓練過程
    for epoch in range(100):
        epoch_loss = 0
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch + 1) % 10 == 0:
            #     torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch + 1))

            epoch_loss = epoch_loss + loss.item()

        print(f'epoch [{epoch + 1:.3d}], loss:{epoch_loss:.5f}')

    # 訓練完成後儲存 model
    # torch.save(model.state_dict(), './checkpoints/last_checkpoint.pth')
