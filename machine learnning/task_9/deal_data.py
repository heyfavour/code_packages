import numpy as np
from torch.utils.data import Dataset


class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images


def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (8500, 32, 32, 3)
    Returns:
      image_list: List of images (8500, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    print(image_list[0])
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


def deal_data():
    trainX = np.load('trainX.npy')
    trainX = preprocess(trainX)
    img_dataset = Image_Dataset(trainX)
    return img_dataset


if __name__ == '__main__':
    img_dataset = deal_data()
