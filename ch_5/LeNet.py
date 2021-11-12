import time
import torch
from torch import nn, optim
import utils
import sys
sys.path.append("..")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        '''
        conv层进行卷积操作
        第一层从 [256，1，28，28] 到 [256, 6, 24, 24] 经过最大值池化 [256， 6， 12， 12]
        第二层从 [256, 6, 12, 12] 到 [256, 16, 8, 8] 经过最大池化 [256, 16, 4, 4]
        fc层进行全连接操作
        神经元从16*4*4 到 120 到 84 最后到 10 进行 10分类操作。
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def evaluate_accuracy(data_iter, net, device=None):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)  # 此处为训练得到的训练y_hat值
            l = loss(y_hat, y) # 此处让y_hat与y进行损失计算 反向梯度传播
            optimizer.zero_grad() # 梯度清零
            l.backward() # 用计算得到的loss进行反向传播
            optimizer.step() # 更新参数
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LeNet()
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()
