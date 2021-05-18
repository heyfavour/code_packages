import re
import torch
import numpy as np

import torchvision.transforms as transforms
from glob import glob
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            class_idx = int(re.findall(re.compile(r'\d+'), img_path)[0])
            image = Image.open(img_path).resize((256,256))
            image = np.array(image)
            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def get_dataloader(mode='training', batch_size=32):
    dataset = ImageDataset(f'./data/{mode}', transform=trainTransform if mode == 'training' else testTransform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'training'))
    return dataloader


if __name__ == '__main__':
    train_dataloader = get_dataloader('training', batch_size=32)
    valid_dataloader = get_dataloader('validation', batch_size=32)
