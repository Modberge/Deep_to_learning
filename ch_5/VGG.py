import time
import torch
from torch import nn, optim
import utils
import sys
sys.path.append("..")

def vgg_block(num_convs, in_channels, out_channels):
    '''
    vgg_block 提出了一个卷积核的范式
    对于给定的感受野，采用堆积的小卷积核效果由于单个大的卷积核
    增加网络的深度来保证学习更复杂的模式，并且参数更少。
    具体来说对于长度为2n+1的大核 采用n层的3×3小核 效果更佳。
    并且在每一层卷积后 加入Max池化层 2×2 步长2，可以使得网络变小
    '''
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    '''
    VGG网络包括卷积层和全连接层。卷积层模块串联成为数个vgg_block
    超参数由conv_arch定义。
    '''

    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(utils.FlattenLayer(),  # 压平操作 变为将卷积层变为平铺层 进行全连接
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ratio = 8
    fc_features = 512 * 7 * 7 // ratio # c * w * h
    fc_hidden_units = 4096 // ratio # 任意
    small_conv_arch = [(1, 3, 64 // ratio),
                       (1, 64 // ratio, 128 // ratio),
                       (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio),
                       (2, 512 // ratio, 512 // ratio)]
    # 经过了5个 add_module 原本的 224 变成了 1 / 32 为 7
    net = vgg(small_conv_arch, fc_features , fc_hidden_units )
    print(net)
    batch_size = 512
    train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()