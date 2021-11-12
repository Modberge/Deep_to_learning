import time
import torch
from torch import nn, optim
import utils
import sys
import torch.nn.functional as F
sys.path.append("..")

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    '''
     NIN模块封装了一个卷积层和两个1X1卷积层 这相当于卷积层和全连接层进行串行操作
     整体可以看作是AlexNet的改进
    '''
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    '''
    相比AlexNet NiN并没有使用3个全连接层 而是使用了平均池化层
    将4维[样本，通道，高，宽]输出转为了2维输出[样本，特征]
    '''
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0), #54
        nn.MaxPool2d(kernel_size=3, stride=2), #26
        nin_block(96, 256, kernel_size=5, stride=1, padding=2), #26
        nn.MaxPool2d(kernel_size=3, stride=2), #12
        nin_block(256, 384, kernel_size=3, stride=1, padding=1), #12
        nn.MaxPool2d(kernel_size=3, stride=2), #[128,384,5,5]
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, stride=1, padding=1), #[128,10,5,5]
        GlobalAvgPool2d(), #[128,10,1,1]
        # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
        utils.FlattenLayer()) #[128,10]

    batch_size = 128
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()  #TODO:似乎训练cifar10数据集的效果很差