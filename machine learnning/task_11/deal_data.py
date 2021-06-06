import os
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

# hyperparameters
batch_size = 128


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = np.array(Image.open(f"data/{fname}"))  # [96 96 3]
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    image_dir = sorted(os.listdir(root))
    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),  # 归一化
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化
    ])
    dataset = FaceDataset(image_dir, transform)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset('data')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    img = np.array(Image.open(f"./data/0.jpg"))
    print(img.shape)
