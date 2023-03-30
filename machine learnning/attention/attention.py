import torch.nn as nn


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(40, 80),
			nn.ReLU(),
			nn.Linear(80, 160),
			nn.TransformerEncoderLayer(d_model=160, dim_feedforward=256, nhead=2, batch_first=True),
		)
		self.classifier = nn.Sequential(
			nn.Linear(160, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 600),
		)

	def forward(self, input):
		mid = self.encoder(input)
		mid = mid.mean(dim=1)
		out = self.classifier(mid)
		return out