import operator
from functools import partial
from functools import reduce
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utilize.process_data import MatLoader
from utilize.configs import activation_dict

from train_utils.adam import Adam
from other_scripts.utilities4 import *
from utilize.loss_metrics import FieldsLpLoss

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, 128)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(128, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc2 = nn.Linear(self.width, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, grid):
        # grid = self.get_grid(x.shape, x.device)
        x1 = x[:,:,:,:1]
        x = torch.cat((x1, grid), dim=-1)
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        return x

    # 虚拟坐标
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     grid = torch.cat((gridx, gridy), dim=-1).to(device)
    #     return grid


pretrain = True
finetune = not pretrain

ntrain = 600
ntest = 100

batch_size = 32
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
h = int(((128 - 1) / r) + 1)  # 128
s = h

print(s)

# nu   (10000,128,128) ; tensor(10000,1,128,128); x(128,)
file_path = os.path.join(r'D:\pythonProject\FNO_PINN\data\2D_DarcyFlow_beta1.0_Train.hdf5')

reader = MatLoader(file_path)
fields = reader.read_field('tensor').permute(3, 0, 1, 2)  # (128,128,1,10000)
print('s',fields.shape)

coords_x = reader.read_field('x-coordinate')  # (128,)
coords_y = reader.read_field('y-coordinate')

Coords_x, Coords_y = torch.meshgrid(coords_x, coords_y)
coords = torch.stack([Coords_x, Coords_y], dim=2)  # np-->axis   torch-->dim
print(coords.shape)  # (128,128,2)
print(coords.shape)
# coords_tile = torch.tile(coords[:,:,:, None],(1,1,1,10000)).permute(3,0,1,2)   #(128,128,2,1000)
coords_tile = torch.tile(coords[:, :, :, None], (1, 1, 1, 10000)).permute(3, 0, 1, 2)
coords_tile.requires_grad_(True)

nu = reader.read_field('nu')

nu_tile = torch.tile(nu[:, :, None, :], (1, 1, 1, 1)).permute(3, 0, 1, 2)
#
input = torch.concat((nu_tile, coords_tile), dim=-1).cuda()
# input = nu_tile
# coords_tile.requires_grad_(True)


output = fields


train_x = input[:ntrain, ::r, ::r][:, :s, :s]
train_y = output[:ntrain, ::r, ::r][:, :s, :s]

print(torch.mean(train_x), torch.mean(train_y))

myloss = FieldsLpLoss()


# 使用FDM solve darcy中的拉普拉斯算子  （中心有限差分法）
def FDM_Darcy(u, a, D=1, f=1):  # u:fno  a:input
    batchsize = u.size(0)  # 张量u的批量大小
    size = u.size(1)  # 维度大小
    u = u.reshape(batchsize, size, size)  # （batchsize, 空间维度x, 空间维度y）
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)  # 计算格点间的网格间距
    dy = dx

    # ux: (batch, size-2, size-2)   #去除了边界
    dudx = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)  # 中心有限差分，计算相邻格点的差分。（2*dx）表示x方向的有限差分步长。
    dudy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)  # 注意切片方法：取每行的第二个元素到倒数第二个（不包括）--->为了避免处理数组的边界元素。

    dadx = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    dady = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    d2udx2 = (u[:, 2:, 1:-1] - 2 * u[:, 1:-1, 1:-1] + u[:, :-2, 1:-1]) / (dx ** 2)
    d2udy2 = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dy ** 2)

    a = a[:, 1:-1, 1:-1]  # 去除边界上的数据
    u = u[:, 1:-1, 1:-1]

    daudx = a * dudx
    daudy = a * dudy
    d2audx2 = (daudx[:, 2:, 1:-1] - daudx[:, :-2, 1:-1]) / (2 * dx)
    d2audy2 = (daudy[:, 1:-1, 2:] - daudy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (d2audx2 + d2audy2)

    return Du


# train_fno and train_bs 为None
def Fourier_Darcy(u, a, D=1, f=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)

    # 计算速度场在频率空间的表示
    u_h = torch.fft.fftn(u, dim=[-2, -1])  # 在最后两个维度执行傅里叶变换（x,y方向）

    # 构建波数网格
    k_x = a * torch.cat((torch.arange(start=0, end=size // 2, step=1, device=u.device),
                     torch.arange(start=-size // 2, end=0, step=1, device=u.device)), 0).reshape(size, 1).repeat(1,size).reshape(1, size, size)
    # torch.arange(start=0, end=size // 2, step=1, device=u.device)：创建从0到（size/2-1）的张量，表示正频率。
    # torch.arange(start=-size // 2, end=0, step=1, device=u.device)), 0：表示负频率
    # cat在0维度拼接后得到完整的正负频率范围。
    k_y = a * torch.cat((torch.arange(start=0, end=size // 2, step=1, device=u.device),
                     torch.arange(start=-size // 2, end=0, step=1, device=u.device)), 0).reshape(1, size).repeat(size,1).reshape(1, size, size)
    # reshape(1, size)：改变形状，使每个频率为列向量。.repeat(size,1)：重复列向量，使每一列都变成相同频率。reshape(1, size, size)：重塑为3d张量，每一层表示一个频率。
    # 在k_y中创建一个具有相同频率的二维网格。
    # 计算负拉普拉斯算子在频率空间的表示
    lap = - D * (k_x ** 2 + k_y ** 2)
    # 计算速度场在频率空间的表示
    u_h1 = u_h * lap
    # 逆傅里叶变换，转回实空间
    Du = torch.fft.ifftn(u_h1, dim=[-2, -1]).real

    return Du


def Auto_Darcy(u, a, coords_tile):

    # 为什么u梯度为空？
    # u = torch.tensor(u, requires_grad=True, dtype=torch.float32)
    # print('uuuu', u.shape)
    # # coords_tile = torch.tensor(coords_tile, requires_grad=True, dtype=torch.float32)
    # coords_tile.requires_grad = True
    # a = torch.tensor(a, requires_grad=True, dtype=torch.float32)

    # coords_x = torch.tensor(coords_tile[:32, :, :, 0], requires_grad=True, dtype=u.dtype)
    # coords_y = torch.tensor(coords_tile[:32, :, :, 1], requires_grad=True, dtype=u.dtype)
    coords_x = coords_tile[:32, :, :, 0]
    coords_y = coords_tile[:32, :, :, 1]

    print("u", u.sum().shape)

    dudx = torch.autograd.grad(u, coords_x, create_graph=True, allow_unused=True, retain_graph=True,
                             only_inputs=True)[0]
    dudy = torch.autograd.grad(u, coords_y, create_graph=True, allow_unused=True, retain_graph=True,
                             only_inputs=True)[0]

    print("dudx", dudx)
    print("dudy", dudy)

    daudx = a * dudx
    daudy = a * dudy
    d2audx2 = torch.autograd.grad(daudx.sum(), coords_x, create_graph=False)[0]
    d2audy2 = torch.autograd.grad(daudy.sum(), coords_y, create_graph=False)[0]
    Du = - (d2audx2 + d2audy2)
    return Du


# PINO: 提供了方程损失、边界条件损失  FNO：提供数据损失
def PINO_loss(u, a, coords_tile):  # u是fno的预测值，a是input即数据真实值
    batchsize = u.size(0)


    size = u.size(1)
    # print('uu',u.shape)
    # print('aa',a.shape)
    u = u.reshape(batchsize, size, size)

    a = a.reshape(batchsize, size, size)
    # coords_tile = coords_tile.reshape(batchsize, size, size)
    # print('a.shpe',a.shape)
    coords_x = coords_tile[:,:,:,0]
    coords_y = coords_tile[:,:,:,1]
    lploss = FieldsLpLoss()

    # 构建上、下、左、右边界索引
    # index_x = torch.cat(
    #     [torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size - 1, 1, -1)),
    #      torch.zeros(size)], dim=0).long()  # （边界左侧；边界上侧；边界右侧；边界下侧）
    # index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size - 1, 1, -1)), torch.zeros(size),
    #                      torch.tensor(range(0, size))], dim=0).long()  # （边界上侧；边界右侧；边界下侧；边界左侧）

    index_x = coords_x.long()
    index_y = coords_y.long()

    boundary_u = u[:, index_x, index_y]  # 提取了u在边界上的值
    truth_u = torch.zeros(boundary_u.shape, device=u.device)  # 与u相同的全0数组-->达西流的边界条件（第一类边界条件）
    loss_bd = lploss.abs(boundary_u, truth_u)  # 计算fno预测与真实值的绝对误差

    Du = FDM_Darcy(u, a)  # 计算了u and a的拉普拉斯算子
    equation = torch.ones(Du.shape, device=u.device)
    loss_equ = lploss.rel(Du, equation)

    return loss_equ, loss_bd  # equation loss; boundary loss.


error = np.zeros((epochs, 4))
#
# coords = coords.cuda()
# mollifier = torch.sin(np.pi*coords[:,:,:,:coords.shape[-1]//2]) * torch.sin(np.pi*coords[:,:,:,coords.shape[-1]//2:]) * 0.001    #sin(pai*x)sin(pai*y)    zero boundary conditions

# print(mollifier.shape)

# 是否使用预训练模型
if pretrain:
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size,
                                               shuffle=True)

    # 使用FNO2d初始化模型
    model = FNO2d(modes, modes, width).cuda()
    # model = FNO2d(in_dim=train_x.shape[-1], out_dim=train_y.shape[-1], modes=(32, 8), width=32, depth=4).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for iteration in range(epochs):
        model.train()
        t1 = default_timer()
        train_pino = 0
        train_fno = 0
        train_loss = 0
        # 遍历训练集
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()  # x: input of training set; y: output of the truth
            optimizer.zero_grad()

            x1 = x[:,:,:,:1]
            coords_tile = x[:,:,:,-2:]
            out = model(x1, coords_tile)  # x: the input of model
            # out = out * mollifier
            # 计算PINN loss
            loss_data = myloss(out, y)  # compute the output of FNO predicted and true data
            loss_equ, loss_bd = PINO_loss(out, x1, coords_tile)
            pino_loss = loss_equ
            loss_batch = 1 * loss_equ + 1 * loss_bd + 10 * loss_data

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            train_fno += loss_data.item()
            train_pino += pino_loss.item()
            train_loss += torch.tensor([loss_equ, loss_bd])

        scheduler.step()

        print('------iteration-------', iteration)
        print('train_pino: {:.3e}, train_fno: {:.3e}'.
              format(train_pino, train_fno))
        print('train_loss', train_loss)

        if iteration % 2 == 0:
            fig, axs = plt.subplots(1, 3, figsize=(8, 8))
            axs[0].imshow(y[2, :, :, 0].detach().cpu().numpy())
            axs[1].imshow(out[2, :, :, 0].detach().cpu().numpy())
            axs[2].imshow(abs(out - y)[2, :, :, 0].detach().cpu().numpy())
            plt.title('true:left  pred:rigt')
            plt.show()

    # torch.save(model, '../model/IP-dracy-forward')

#############
# NOTE:
# 1、FNO的data loss加入总loss，权重分配------>
# 3、边界条件需要加上，当2的coords解决后



