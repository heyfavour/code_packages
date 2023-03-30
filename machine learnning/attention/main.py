import math
import torch
import torch.nn as nn

from torch.optim import AdamW
from deal_data import get_dataloader
from attention import Classifier
from transformers import optimization


if __name__ == '__main__':
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	train_loader, valid_loader, speaker_num = get_dataloader()#
	print(f"[Info]: Finish loading data!",flush = True)

	model = Classifier().to(device)#600
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=1e-4)
	epoch_num = 500
	scheduler = optimization.get_cosine_schedule_with_warmup(optimizer, 1000, 1593*epoch_num)

	print(f"[Info]: Finish creating model!",flush = True)

	for epoch in range(epoch_num):#1593*epochs
		for idx,(mel,speaker) in enumerate(train_loader):
			mel,speaker = mel.to(device),speaker.to(device)
			output = model(mel)
			loss = criterion(output, speaker)

			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

			accuracy = torch.mean((output.argmax(1) == speaker).float()).item()
			print(f"[EPOCH {epoch:0>4d}/{epoch_num:0>4d}][NUM {idx:0>4d}][{accuracy:.4f}][LOSS {loss.sum().item():.4f}]")
		#valid_accuracy = valid(valid_loader, model, criterion, device)
