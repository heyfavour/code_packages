import torch.nn as nn

class Generator(nn.Module):
  """
  input (N, in_dim)
  output (N, 3, 64, 64)
  """
  def __init__(self, in_dim, dim=64):
    super(Generator, self).__init__()
    self.l1 = nn.Sequential(
      nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),#100 64 5 2 2
      nn.BatchNorm1d(dim * 8 * 4 * 4),
      nn.ReLU()
    )
    self.l2_5 = nn.Sequential(
      nn.Sequential(
        nn.ConvTranspose2d(dim * 8, dim * 4, 5, 2,padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(dim * 4),
        nn.ReLU()
      ),
      nn.Sequential(
        nn.ConvTranspose2d(dim * 4, dim * 2, 5, 2,padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(dim * 2),
        nn.ReLU()
      ),
      nn.Sequential(
        nn.ConvTranspose2d(dim * 2, dim * 1, 5, 2,padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(dim * 1),
        nn.ReLU()
      ),
      nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
      nn.Tanh()
    )

  def forward(self, x):
    y = self.l1(x)
    y = y.view(y.size(0), -1, 4, 4)
    y = self.l2_5(y)
    return y

if __name__ == '__main__':
    from torchsummary import summary
    model = Generator(100).cuda()
    summary(model,input_size = (128,100))
