def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)
        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq
        return activation(ret)  # activation

import torch
import torchvision
import time
from torch import nn
import torch.nn.functional as F

#---------------------------DataProcess---------------------------------------------------------------------------

def load_data_fashion_mnist(batch_size, resize=None, root='./Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter


def load_data_cifar10(batch_size, resize=None, root='./Datasets/CIFAR10'):
    """Download the CIFAR10 dataset and then load into memory."""
    trans = []
    if resize:  # 设定将原数据图像扩大到resize的大小
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    cifar10_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter

#--------------------------Train-and-Evaluate---------------------------------------------------------------------

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
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
        if epoch == num_epochs-1:
            test_acc = evaluate_accuracy(test_iter, net, device)
            print('train finish','\n','test acc %.4f' %(test_acc))
            return test_acc
#----------------------------fun-----------------------------------
def corr2d(in_array,kernel):

    k_1,k_2 = kernel.shape

    out_array = torch.zeros(
        in_array.shape[0]-k_1+1,
        in_array.shape[1]-k_2+1,
    )
    for i in range(out_array.shape[0]):
        for j in range(out_array.shape[1]):
            out_array[i,j] = (in_array[i:i+k_1,j:j+k_2]*kernel).sum()
    return out_array

def comp_conv2d(conv2d, X):
    # (1,1)代表批量大小和通道数
    X = X.view((1,1)+X.shape)
    Y = conv2d(X)

    return Y.view(Y.shape[2:]) #截取1，1之后的shape

def corr2d_multi_in(X,kernel): #多输入通道互相关运算，通过累加函数计算

    res = torch.zeros(
        (X.shape[1]-kernel.shape[1]+1,
         X.shape[2]-kernel.shape[2]+1)
    )
    for i in range(0, X.shape[0]):
        res += corr2d(X[i,:,:], kernel[i,:,:])
    return res

def corr2d_multi_in_out(X,kernel):
    # 如果说X是n维空间
    # 在多通道输入单通道输出时输出为n-1维空间 核函数也是n维空间
    # 在多通道输入多通道输出时输出为n维空间 核函数应该为n+1维空间
    # 其中第一维是输出维度 第二维是输入维度
    return torch.stack([corr2d_multi_in(X,k) for k in kernel])

def corr2d_multi_in_out_1x1(X,kernel):
    """
    把通道维度看作全连接网络特征维，高和宽看作数据样本
    则1*1卷积层相当于全链接层。
    卷积层的第二维度即为特征维度
    卷积层的第一维度即为下一个连接层维度
    """
    c_1, h, w = X.shape
    c_0 = kernel.shape[0]
    X = X.view(c_1, h * w)
    kernel = kernel.view(c_0, c_1)
    Y = torch.mm(kernel, X)  # 全连接层的矩阵乘法
    return Y.view(c_0, h, w)

def pool2d(X, pool_size, mode='max'):
    '''
    池化层的输出通道和输入通道数相同
    池化层的填充和步幅可以进行补充
    池化层的一个主要作用是缓解卷积层对位置的过度敏感性
    '''
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i: i + p_h, j: j + p_w].max()
            if mode == 'avg':
                Y[i,j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

#----------------------------class-------------------------------

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Conv2D(nn.Module): #二维卷积层

    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self,x):
        return corr2d(x,self.weight)+self.bias



class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])



#----------------------------run---------------------------------
def class5_1():

    conv2d = Conv2D(kernel_size=(1,2))

    X = torch.ones(6,8)
    X[:,2:6] = 0

    K = torch.tensor([[1,-1]])

    Y = corr2d(X,K)

    step = 20
    lr = 0.01

    for i in range(step):
        Y_hat = conv2d(X)
        l = ((Y_hat - Y)**2).sum()
        l.backward()

        conv2d.weight.data -= lr*conv2d.weight.grad
        conv2d.bias.data -= lr*conv2d.bias.grad

        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)

        if(i+1) % 5 ==0:
            print('Setp %d, loss %.3f' %(i+1, l.item()))

def class5_2():



    conv2d = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
    X = torch.rand(8, 8)
    shape = comp_conv2d(conv2d, X).shape
    print(shape)

def class5_3():
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

    K = torch.stack([K, K + 1, K + 2])

    out = corr2d_multi_in_out(X, K)
    print(out)

    X = torch.rand(3, 3, 3)
    K = torch.rand(2, 3, 1, 1)

    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)

    print((Y1 - Y2).norm().item() < 1e-6)

def class5_4():
    X = torch.tensor([[0,1,2], [3,4,5], [6,7,8]])
    X = pool2d(X, (2, 2),'avg')

    X = torch.arange(16, dtype=torch.float).view((1,1,4,4))
    pool_1 = nn.MaxPool2d(3)
    X_1 = pool_1(X)
    pool_2 = nn.MaxPool2d(3, padding=1, stride=2)
    X_2 = pool_2(X)
    pool_3 = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
    X_3 = pool_3(X)

    X = torch.cat((X, X + 1), dim=1)
    pool_4 = nn.MaxPool2d(3, padding=1, stride=2)
    X_4 = pool_4(X)
    print(X_4)




if __name__ == "__main__" :
    class5_4()

