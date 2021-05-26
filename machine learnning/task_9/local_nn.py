import random
import torch.nn as nn
import numpy as np

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import torch
    from torch import optim

    same_seeds(0)

    model = AE().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    model.train()
    n_epoch = 100

    # 準備 dataloader, model, loss criterion 和 optimizer
    img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

    epoch_loss = 0

    # 主要的訓練過程
    for epoch in range(n_epoch):
        epoch_loss = 0
        for data in img_dataloader:
            img = data
            img = img.cuda()

            output1, output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch + 1))

            epoch_loss += loss.item()

        print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, epoch_loss))

    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './checkpoints/last_checkpoint.pth')
