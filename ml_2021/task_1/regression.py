import torch
import torch.nn as nn
from deal_data import get_dataloader


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(93, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input):
        ouput = self.net(input)
        return ouput


if __name__ == '__main__':
    dataloader = get_dataloader()
    model = NeuralNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(30000):
        _loss = 0
        for i, (input, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(input)  #
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            _loss = _loss + loss.item()
        print(outputs[:5], labels[:5])
        print(f"epoch:{epoch} loss:{_loss / (i + 1)}")
