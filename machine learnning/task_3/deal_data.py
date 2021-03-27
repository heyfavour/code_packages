import os
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import func_log

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

validation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, X, Y, transform):
        self.X = X
        self.Y = torch.LongTensor(Y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.transform(self.X[index])
        Y = self.Y[index]
        return X, Y


def read_pictures(path):
    pictures = os.listdir(path)
    train_x = []
    train_y = []
    for index, pic in enumerate(pictures):
        image = Image.open(f"{path}/{pic}").resize((128, 128))
        image = np.array(image)
        train_x.append(image)
        label = pic.split("_")[0]
        train_y.append(int(label))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print(f"load {path} data {len(train_x)}")
    return train_x, train_y


@func_log
def train_data_loader():
    path = "training"
    data_x, data_y = read_pictures(path)
    train_dataset = ImgDataset(data_x, data_y, train_transform)  # data = Dataset(data_x, data_y)
    loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    return loader, len(data_x)


@func_log
def validation_data_loader():
    path = "validation"
    data_x, data_y = read_pictures(path)
    validation_dataset = ImgDataset(data_x, data_y, validation_transform)
    loader = DataLoader(dataset=validation_dataset, batch_size=128, shuffle=True)
    return loader, len(data_x)

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        #img = cv2.imread(os.path.join(path, file))
        #x[i, :, :] = cv2.resize(img,(128, 128))
        x[i, :, :] = Image.open(os.path.join(path,file)).resize((128,128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x


if __name__ == '__main__':
    path = "validation"
    data_x, data_y = read_pictures(path)
    x1,y1 = readfile(path,True)
    print(data_x.shape)
    print(x1.shape)
    print(data_x==x1)
