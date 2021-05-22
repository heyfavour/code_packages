import torch.nn as nn
import torch.nn.functional as F
import torch


class StudentNet(nn.Module):
    '''
    在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
    你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。
    另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]
        bandwidth = [base * m for m in multiplier]  # bandwidth: 每一層Layer所使用的channel數量 [16, 32, 64, 128, 256, 256, 256, 256]
        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7): bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # Conv2d input output kernel_size stride padding group
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),#[3,m,m]=>[16,m,m]
                nn.BatchNorm2d(bandwidth[0]),#[16,m,m]
                nn.ReLU6(),#[16,m,m]
                nn.MaxPool2d(2, 2, 0),#[16,m,m]
            ),
            # 第一层Convolution Layer
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),#[16,m,m]
                # Depthwise  Depthwise卷积后再加ReLU效果变差
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),  # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),  #[32 16 1 1]# Pointwise 經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),  # Depthwise[16 m m ]
                nn.BatchNorm2d(bandwidth[1]),#[16 m m ]
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),  # Pointwise []
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),  # Depthwise
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),  # Pointwise
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),
            nn.AdaptiveAvgPool2d((1, 1)),  # 池化成 1*1的图片
        )
        self.fc = nn.Sequential(nn.Linear(bandwidth[7], 11), )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

if __name__ == '__main__':
    StudentNet(base=16)
