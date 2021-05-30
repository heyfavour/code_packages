import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from deal_data import deal_data, preprocess
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans


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
        code = self.encoder(x)  # [64 256 4 4]
        output = self.decoder(code)
        return code, output


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


def train_model():
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

            epoch_loss = epoch_loss + loss.item()

        print(f'epoch [{epoch + 1:0=3d}], loss:{epoch_loss:.5f}')

    # 訓練完成後儲存 model
    torch.save(model.state_dict(), './last_checkpoint.pth')


def inference(train_x, model, batch_size=256):
    latents = []
    for i, x in enumerate(train_x):
        x = torch.FloatTensor(x)  # [64 3 32 32]
        code, output = model(x.cuda())  # [64 256 4 4] [64 3 32 32]
        if i == 0:
            # outpu t.size()[0]=batch_size 64  =>[64 256*4*4]
            latents = code.view(output.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, code.view(output.size()[0], -1).cpu().detach().numpy()), axis=0)
    print('Latents Shape:', latents.shape)
    return latents  # [8500 256*4*4]


def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)  # [8500 200]
    print('First Reduction Shape:', kpca.shape)

    # Second Dimesnion Reduction
    tsne_x = TSNE(n_components=2).fit_transform(kpca)  # [8500 2]
    print('Second Reduction Shape:', tsne_x.shape)
    # Clustering
    predict_y = MiniBatchKMeans(n_clusters=2, random_state=0).fit(tsne_x)
    predict_y = [int(i) for i in predict_y.labels_]
    predict_y = np.array(predict_y)  # 8500 [1 0 ...]
    return predict_y, tsne_x


def save_prediction(pred, out_csv='prediction.csv'):
    # pred = np.abs(1 - pred) #避免聚类分成相反方向
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')


def cal_acc(gt, pred):
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return max(acc, 1 - acc)


def plot_scatter(feat, label):
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c=label)
    plt.legend(loc='best')
    plt.show()
    return


if __name__ == '__main__':
    same_seeds(0)

    model = AutoEencoder().cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    model.train()
    img_dataloader = deal_data(data_set="trainX.npy")
    # train_model()
    model.load_state_dict(torch.load('./last_checkpoint.pth'))
    model.eval()

    # 預測答案
    # latents = inference(train_x=img_dataloader, model=model)  # latents [8500 256*4*4] [index code]
    # predict_y, tsne_x = predict(latents)
    # save_prediction(predict_y, 'prediction.csv')

    # 聚类标点图
    # valX = deal_data(data_set="valX.npy")
    # valY = np.load('valY.npy')
    # latents = inference(valX, model)
    # predict_y, tsne_x = predict(latents)  # 聚类分析
    # acc_latent = cal_acc(valY, predict_y)
    # print('The clustering accuracy is:', acc_latent)
    # plot_scatter(tsne_x, valY)

    # 畫出原圖
    plt.figure(figsize=(10, 4))
    indexes = [1, 2, 3, 6, 7, 9]
    trainX = np.load('trainX.npy')
    trainX_preprocessed = preprocess(trainX)
    imgs = trainX[indexes,]
    for i, img in enumerate(imgs):
        plt.subplot(2, 6, i + 1, xticks=[], yticks=[])
        plt.imshow(img)
    # 畫出 reconstruct 的圖
    inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
    latents, recs = model(inp)
    recs = ((recs + 1) / 2).cpu().detach().numpy()
    recs = recs.transpose(0, 2, 3, 1)
    for i, img in enumerate(recs):
        plt.subplot(2, 6, 6 + i + 1, xticks=[], yticks=[])
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
