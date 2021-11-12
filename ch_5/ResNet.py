import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
import sys
sys.path.append("..")

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def get_net():
    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", utils.GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(utils.FlattenLayer(), nn.Linear(512, 10)))
    return net

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, lr, num_epochs, decay =256, 0.001, 15, 1e-3

    list_batch_size = [32,64,128,256,512]
    list_lr = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    list_num_epochs = [1, 5, 10, 15, 20]
    list_decay = [1, 1e-1, 1e-2, 1e-3, 1e-4]

    for batch_size in list_batch_size:
        with open("result.txt","a") as f:
            f.write('start change batch_size'+'\n')
        net = get_net()
        train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=96)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
        test_acc = utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        with open("result.txt","a") as f:
            f.write('batch size %d, test acc %.4f' %(batch_size,test_acc)+'\n')

    for lr in list_lr:
        with open("result.txt","a") as f:
            f.write('start change lr','\n')
        net = get_net()
        train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=96)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
        test_acc = utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        with open("result.txt","a") as f:
            f.write('lr %f, test acc %.4f' %(lr, test_acc),'\n')

    for num_epochs in list_num_epochs:
        with open("result.txt","a") as f:
            f.write('start change num epcohs','\n')
        net = get_net()
        train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=96)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
        test_acc = utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        with open("result.txt","a") as f:
            f.write('num epochs %d, test acc %.4f' %(num_epochs,test_acc),'\n')

    for decay in list_decay:
        with open("result.txt","a") as f:
            f.write('start change decay','\n')
        net = get_net()
        train_iter, test_iter = utils.load_data_cifar10(batch_size, resize=96)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=decay)
        test_acc = utils.train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
        with open("result.txt","a") as f:
            f.write('decay %f, test acc %.4f' %(decay,test_acc),'\n')

if __name__ == '__main__':
    main()









    