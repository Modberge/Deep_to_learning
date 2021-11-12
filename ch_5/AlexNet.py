import time
import torch
from torch import nn, optim
import torchvision
import utils
import sys
sys.path.append("..")

class AlexNet(nn.Module):
    '''
    端到端的学习

    首次证明了学习的特征可以超过手工设计的特征

    意义：
    AlexNet跟LeNet结构类似，但使用了更多的卷积层和
    更大的参数空间来拟合大规模数据集ImageNet。
    它是浅层神经网络和深度神经网络的分界线。
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), # 为了捕捉更大的特征 54
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride 26
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2), # 26
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # 12
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1), # 12
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1), # 12
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1), # 12
            nn.ReLU(),
            nn.MaxPool2d(3, 2) # 5
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # 控制模型复杂度
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AlexNet()
    batch_size = 256
    train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=224)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()