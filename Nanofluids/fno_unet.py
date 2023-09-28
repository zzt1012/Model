#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/1 14:38
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_train_FNO.py
"""
import h5py
import pandas as pd
import scipy
from scipy import io
from torch.utils.data import DataLoader
# from torchinfo import summary
from FNO.FNOs import FNO2d
from utilize.loss_metrics import FieldsLpLoss
from utilize.process_data import DataNormer, MatLoader
from CNN.ConvNets import UNet2d, DownSampleNet2d, UpSampleNet2d
from utilize.visual_data import MatplotlibVision, TextLogger
# from FUL import CnnNet

import matplotlib.pyplot as plt
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary


# 这个函数可以用来生成输入图像的坐标网格，并与其他特征进行拼接，以提供位置信息。这可以帮助生成网络学习生成具有正确位置和结构的图像。
# 特征变换函数:输入x坐标转换为输入res
def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    # print("Shape of x:", x.shape)  #torch.Size([32, 160])
    # print("Shape of batchsize:", shape[0])   #
    # print("Shape of size_x:", shape[1])

    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 0.005, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 0.0003, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)  # 沿着将 gridx 和 gridy 张量沿着最后一个维度进行拼接，
    # 并将拼接后的张量移动到与输入张量 x 相同的设备上。拼接后的张量作为输出返回。


def train(dataloader, netmodel, device, lossfunc, fieldloss, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    train_loss = 0
    train_floss_p = 0
    train_floss_t = 0
    train_floss_u= 0
    train_floss_v = 0
    train_floss_total=0

    train_floss_pm = 0
    train_floss_tm = 0
    train_floss_um = 0
    train_floss_vm = 0
    train_floss_totalm = 0


    # batch_size=42
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)

        # gd = feature_transform(xx)
        gd = xx[:, :, :, -2:]
        xx1 = xx[:, :, :, :8]

        input_size = list(xx1.shape)

        pred = netmodel(xx1, gd)  # 使用网络模型netmodel特征变换后的数据进行前向传播
        loss = lossfunc(pred, yy)
        #field_floss= fieldloss(pred.reshape(input_size[0],-1,1), yy.reshape(input_size[0],-1,1))
# #取最后一个样本的fieldloss,每次控制一个样本
        field_floss_p = fieldloss.abs(pred, yy)[:,0].mean() #rel/abs计算出（32,4）
        field_floss_t =fieldloss.abs(pred, yy) [:,1].mean()
        field_floss_u =fieldloss.abs(pred, yy) [:,2].mean()
        field_floss_v = fieldloss.abs(pred, yy) [:,3].mean()

        field_floss_pm = fieldloss.abs(pred, yy)[:, 0].max()  # rel/abs计算出（32,4）
        field_floss_tm = fieldloss.abs(pred, yy)[:, 1].max()
        field_floss_um = fieldloss.abs(pred, yy)[:, 2].max()
        field_floss_vm = fieldloss.abs(pred, yy)[:, 3].max()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 使用优化器 optimizer的step方法更新模型的参数，根据计算得到的梯度进行优化。
        train_loss += loss.item()
        train_floss_p += field_floss_p.item()
        train_floss_t += field_floss_t.item()
        train_floss_u +=field_floss_u.item()
        train_floss_v+=field_floss_v.item()

        train_floss_pm += field_floss_pm.item()
        train_floss_tm += field_floss_tm.item()
        train_floss_um += field_floss_um.item()
        train_floss_vm += field_floss_vm.item()

        # train_floss_total.append(train_floss_p +train_floss_t +train_floss_u +train_floss_v)
        # train_floss_totalm.append(train_floss_pm + train_floss_tm + train_floss_um + train_floss_vm)

    scheduler.step()

    # return train_loss / (batch + 1) / batch_size, train_floss_total /(batch+1)/batch_size, \
    #        train_floss_totalm /(batch +1)/batch_size

    return train_loss / (batch + 1) / batch_size, train_floss_p / (batch + 1) / batch_size, \
           train_floss_t / (batch + 1) / batch_size, train_floss_u / (batch + 1) / batch_size, train_floss_v / (
                       batch + 1) / batch_size, train_floss_pm / (batch + 1) / batch_size, \
           train_floss_tm / (batch + 1) / batch_size, train_floss_um / (batch + 1) / batch_size, train_floss_vm / (
                       batch + 1) / batch_size,

def valid(dataloader, netmodel, device, lossfunc, fieldloss):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    valid_floss_p = 0
    valid_floss_t = 0
    valid_floss_u = 0
    valid_floss_v = 0
    valid_floss_total= 0

    valid_floss_pm = 0
    valid_floss_tm = 0
    valid_floss_um = 0
    valid_floss_vm = 0
    valid_floss_totalm = 0

    # batch=20
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)  # input
            yy = yy.to(device)  # output
            # gd = feature_transform(xx)
            gd = xx[:, :, :, -2:]
            xx1 = xx[:, :, :, :8]

            input_size = list(xx1.shape)

            pred = netmodel(xx1, gd)  # 使用网络模型netmodel特征变换后的数据进行前向传播
            loss = lossfunc(pred, yy)
           # field_floss = fieldloss(pred.reshape(input_size[0], -1, 1), yy.reshape(input_size[0], -1, 1))
            # #取最后一个样本的fieldloss,每次控制一个样本
         #rel/abs计算出（32,4）

            #最后一个样本的fieldloss, 每次控制一个样本
            field_floss_p = fieldloss.abs(pred, yy)[:, 0].mean()  # rel/abs计算出（32,4）
            field_floss_t = fieldloss.abs(pred, yy)[:, 1].mean()
            field_floss_u = fieldloss.abs(pred, yy)[:, 2].mean()
            field_floss_v = fieldloss.abs(pred, yy)[:, 3].mean()

            field_floss_pm = fieldloss.abs(pred, yy)[:, 0].max()  # rel/abs计算出（32,4）
            field_floss_tm = fieldloss.abs(pred, yy)[:, 1].max()
            field_floss_um = fieldloss.abs(pred, yy)[:, 2].max()
            field_floss_vm = fieldloss.abs(pred, yy)[:, 3].max()

            valid_loss += loss.item()
            valid_floss_p += field_floss_p.item()
            valid_floss_t += field_floss_t.item()
            valid_floss_u += field_floss_u.item()
            valid_floss_v += field_floss_v.item()
            # valid_floss_total.append(valid_floss_p,valid_floss_t,valid_floss_u,valid_floss_v)

            valid_floss_pm += field_floss_pm.item()
            valid_floss_tm += field_floss_tm.item()
            valid_floss_um += field_floss_um.item()
            valid_floss_vm += field_floss_vm.item()
            # valid_floss_totalm.append(valid_floss_pm, valid_floss_tm, valid_floss_um, valid_floss_vm)


    # return valid_loss / (batch + 1) / batch_size, valid_floss_total / (batch+1) /batch_size, \
    #        valid_floss_totalm /(batch+1) /batch_size
    return valid_loss / (batch + 1) / batch_size, valid_floss_p / (batch+1) /batch_size, \
           valid_floss_t /(batch+1) /batch_size, valid_floss_u/(batch+1)/batch_size, valid_floss_v / (batch+1)/batch_size,\
           valid_floss_pm / (batch+1) /batch_size, \
           valid_floss_tm /(batch+1) /batch_size, valid_floss_um/(batch+1)/batch_size, valid_floss_vm / (batch+1)/batch_size,


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        # gd = feature_transform(xx)
        gd = xx[:, :, :, -2:]
        xx1 = xx[:, :, :, :8]
        pred = netmodel(xx1, gd)
        # pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    # return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'UNet'
    work_path = os.path.join('work_new_loss2', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    sys.stdout = TextLogger(filename=os.path.join(work_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    sys.stdout.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    # 将控制台的结果输出到log文件
    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    # ----------------------
    # FNO:     lr:0.001  gamma:0.1  beta:[0.99,0.99]  weight_deacy:1e-9  batchsize:32   【200,300,400】
    # UNet:    ntrain:600,batchszie20,laerning rate0.00001,step300,gamma0.1.   

    #初始学习率 3*10-4  权重3e-5  (unet 文献)
    # ####训练集参数
    ntrain = 338  # 600,1000   ，6000(过拟合)                   #40000,1000
    nvalid = 50  # 100,200     ，1000

    r1 = 2  # 空间坐标x  若误差大可以取间隔2
    r2 = 1
    s1 = int(((794 - 1) / r1) + 1)  # 265
    s2 = int(((40 - 1) / r2) + 1)  # 40

    batch_size = 32  # 一个批次处理20个图片
    epochs = 501
    learning_rate = 0.0001  # 0.001
    scheduler_step = [300, 400, 500]  # first300
    scheduler_gamma = 0.1  # 0.1   不加train_x归一化时gamma是0.5   gammm大于0.1就过拟合

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################
    file_path = os.path.join('data', 'dim_pro8_single_all.mat')
    reader = MatLoader(file_path)
    fields = reader.read_field('field')
    # fields=fields[:,::2,:,:]  #第一、三、四个维度不变，第二个维度缩小一半
    design = reader.read_field('data')
    selected_columns = [col for col in range(design.shape[1]) if col not in [1, 8]]
    design = design[:, selected_columns]
    coords = reader.read_field('grids')[..., :2]
    target = torch.concat((reader.read_field('Nu'), reader.read_field('f')), dim=-1)
    design_tile = torch.tile(design[:, None, None, :], (1, 792, 40, 1))
    input_size = design_tile.shape[1:]
    # layer = UNet2d(in_sizes=input_size, out_sizes=fields.shape[1:], width=32, depth=6)

    design_shape = design.shape[-1]
    fields_shape = fields.shape[-1]
    coords_shape = coords.shape[-1]
    target_shape = target.shape[-1]
    design_tile_s = design_tile.shape[1]  # 792

    # 原始场shape(6773,792,40,4)
    # batch_size,coords_x,coords_y,channel=fields.shape[0],fields.shape[1],fields.shape[2],fields.shape[3]

    # 输入10+2，forward传入(x，grid),x->12,grid->2

    input = torch.concat((design_tile, coords), dim=-1)
    output = fields
    # input = torch.tensor(input, dtype=torch.float)
    # output = torch.tensor(fields, dtype=torch.float)
    print(input.shape, output.shape)
    sys.stdout.info("input sizes: {}, output sizes: {}".format(input.shape, output.shape))

    # Visual = MatplotlibVision(os.path.join('work', 'visual'), input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))
    # fig, axs = plt.subplots(4, 3, figsize=(25, 6), num=1)
    # Visual.plot_fields_ms(fig, axs, real=fields[0].numpy(), pred=fields[0].numpy(), coord=coords[0].numpy())
    # plt.show()

    del coords, design
    # train_x   (6000,264,40,12)  train_y  (6000,264,40,4)
    # input两个通道，第一个通道设计变量，第二个空间坐标，取样时要保证两个通道的样本数量相同
    train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # valid_x   (773,264,40,12)  valid_y (773,264,40,4)
    valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]

    del reader
    x_normalizer = DataNormer(train_x.numpy(),
                              method='min-max')  # 本设计无法采用线性化归一化方式，会导致奇异化    train_x = x_normalizer.norm(train_x)
    x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    # print(train_x)

    y_normalizer = DataNormer(train_y.numpy(), method='min-max')
    y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    # print(train_y)

    # shuffle=True表示在每个训练周期（epoch）开始时对数据进行洗牌，以增加数据的随机性。
    # 通过torch.utils.data.TensorDataset将训练集数据train_x和train_y（输入和目标）打包成一个数据集
    # 使用torch.utils.data.DataLoader来创建数据加载器
    # drop_last=True表示如果数据集的大小不能被batch_size整除，则丢弃最后一个不完整的批次。
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    train_x = train_x[:, :, :, :8]
    if name == 'UpSample':
        Net_model = UpSampleNet2d(train_x.shape[-1], out_sizes=train_y.shape[1:], width=32, depth=4).to(Device)
    elif name == 'FNO':
        Net_model = FNO2d(in_dim=train_x.shape[-1], out_dim=train_y.shape[-1], modes=(32, 8), width=32, depth=4).to(
            Device)
    # unet不涉及steps。因为只有非定常问题才会涉及steps
    elif name == 'UNet':
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:], width=32, depth=6).to(Device)
    elif name == 'DownSample':
        Net_model = DownSampleNet2d(in_sizes=train_y.shape[1:], out_sizes=target_shape, width=32, depth=4).to(Device)

    input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[-1]).to(Device)
    input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
    summary(Net_model, input_data=[input1, input2], device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    Error_func = FieldsLpLoss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法  #unet wweight 5e-9
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-13)  # 1e-9  #1e-8(work_new)  1e-10  1e-13(work_new_loss)  beta 0.7 0.9
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))

    star_time = time.time()
    log_loss = [[], []]
    log_floss_p= [[],[]]
    log_floss_t= [[],[]]
    log_floss_u=[[],[]]
    log_floss_v=[[],[]]
    log_floss_pm= [[],[]]
    log_floss_tm= [[],[]]
    log_floss_um=[[],[]]
    log_floss_vm=[[],[]]



    log_floss_total=[[],[]]
    log_floss_totalm=[[],[]]

    ################################################################
    # train process
    ################################################################

    for epoch in range(epochs):

        # Net_model.train()
        # train_loss, trian_floss_total, train_floss_totalm = \
        #     train(train_loader, Net_model, Device, Loss_func, Error_func, Optimizer, Scheduler)
        # log_loss[0].append(train_loss)
        # log_floss_total[0].append(trian_floss_total)
        # log_floss_totalm[0].append(log_floss_totalm)

        Net_model.train()
        train_loss, trian_floss_p, trian_floss_t, train_floss_u, train_floss_v, \
        trian_floss_pm, trian_floss_tm, train_floss_um, train_floss_vm = \
            train(train_loader, Net_model, Device, Loss_func, Error_func, Optimizer, Scheduler)
        log_loss[0].append(train_loss)
        log_floss_p[0].append(trian_floss_p)
        log_floss_t[0].append(trian_floss_t)
        log_floss_u[0].append(train_floss_u)
        log_floss_v[0].append(train_floss_v)

        log_floss_pm[0].append(trian_floss_pm)
        log_floss_tm[0].append(trian_floss_tm)
        log_floss_um[0].append(train_floss_um)
        log_floss_vm[0].append(train_floss_vm)

        Net_model.eval()

        valid_loss, valid_floss_p, valid_floss_t, valid_floss_u, valid_floss_v,\
        valid_floss_pm, valid_floss_tm, valid_floss_um, valid_floss_vm = \
            valid(valid_loader, Net_model, Device, Loss_func, Error_func)
        log_loss[1].append(valid_loss)

        log_floss_p[1].append(valid_floss_p)
        log_floss_t[1].append(valid_floss_t)
        log_floss_u[1].append(valid_floss_u)
        log_floss_v[1].append(valid_floss_v)

        log_floss_pm[1].append(valid_floss_pm)
        log_floss_tm[1].append(valid_floss_tm)
        log_floss_um[1].append(valid_floss_um)
        log_floss_vm[1].append(valid_floss_vm)

        # valid_loss, valid_floss_total, valid_floss_totalm = \
        #     valid(valid_loader, Net_model, Device, Loss_func, Error_func)
        # log_loss[1].append(valid_loss)
        # log_floss_total[1].append(valid_floss_total)
        # log_floss_totalm[1].append(valid_floss_totalm)


        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))
        star_time = time.time()

        if epoch > 0 and epoch % 1 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

            x = np.arange(len(log_loss[0]))
            train_step = np.array(log_loss)[0, :]
            valid_step = np.array(log_loss)[1, :]
            df = pd.DataFrame({'Index': x, 'train_step': train_step, 'valid_step': valid_step})
            writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work_new_loss2\UNet\338_50\loss.xlsx')
            df.to_excel(writer, index=False)
            writer.save()


            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=2)
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_p)[0,:], 'trian_loss_p')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_p)[1, :], 'valid_loss_p')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_t)[0, :], 'train_loss_t')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_t)[1, :], 'valid_loss_t')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_u)[0, :], 'trian_loss_u')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_u)[1, :], 'valid_loss_u')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_v)[0, :], 'trian_loss_v')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_v)[1, :], 'valid_loss_v')
            fig.suptitle('field loss')
            fig.savefig(os.path.join(work_path, 'field_loss_mean.svg'))
            plt.close(fig)

            x = np.arange(len(log_floss_p[0]))
            train_loss_p = np.array(log_floss_p)[0,:]
            valid_loss_p = np.array(log_floss_p)[1, :]
            train_loss_t = np.array(log_floss_t)[0, :]
            valid_loss_t = np.array(log_floss_t)[1, :]
            train_loss_u = np.array(log_floss_u)[0, :]
            valid_loss_u = np.array(log_floss_u)[1, :]
            train_loss_v = np.array(log_floss_v)[0, :]
            valid_loss_v = np.array(log_floss_v)[1, :]
            df = pd.DataFrame({'Index': x, 'train_loss_p': train_loss_p, 'valid_loss_p': valid_loss_p,
                               'train_loss_t': train_loss_t, 'valid_loss_t': valid_loss_t,
                               'train_loss_u': train_loss_u, 'valid_loss_u': valid_loss_u,
                               'train_loss_v': train_loss_v, 'valid_loss_v': valid_loss_v,})
            writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work_new_loss2\UNet\338_50\field_loss_mean.xlsx')
            df.to_excel(writer, index=False)
            writer.save()

            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=3)
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_pm)[0, :], 'trian_loss_pm')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_pm)[1, :], 'valid_loss_pm')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_tm)[0, :], 'train_loss_tm')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_tm)[1, :], 'valid_loss_tm')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_um)[0, :], 'trian_loss_um')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_um)[1, :], 'valid_loss_um')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_vm)[0, :], 'trian_loss_vm')
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_pm[0])), np.array(log_floss_vm)[1, :], 'valid_loss_vm')
            fig.suptitle('field loss')
            fig.savefig(os.path.join(work_path, 'field_loss_max.svg'))
            plt.close(fig)

            x = np.arange(len(log_floss_pm[0]))
            train_loss_pm = np.array(log_floss_pm)[0, :]
            valid_loss_pm = np.array(log_floss_pm)[1, :]
            train_loss_tm = np.array(log_floss_tm)[0, :]
            valid_loss_tm = np.array(log_floss_tm)[1, :]
            train_loss_um = np.array(log_floss_um)[0, :]
            valid_loss_um = np.array(log_floss_um)[1, :]
            train_loss_vm = np.array(log_floss_vm)[0, :]
            valid_loss_vm = np.array(log_floss_vm)[1, :]
            df = pd.DataFrame({'Index': x, 'train_loss_pm': train_loss_pm, 'valid_loss_pm': valid_loss_pm,
                               'train_loss_tm': train_loss_tm, 'valid_loss_tm': valid_loss_tm,
                               'train_loss_um': train_loss_um, 'valid_loss_um': valid_loss_um,
                               'train_loss_vm': train_loss_vm, 'valid_loss_vm': valid_loss_vm, })
            writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work_new_loss2\UNet\338_50\field_loss_max.xlsx')
            df.to_excel(writer, index=False)
            writer.save()

            # fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=2)
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[0, 0], 'trian_loss_p')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[1, 0], 'valid_loss_p')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[0, 1], 'train_loss_t')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[1, 1], 'valid_loss_t')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[0, 2], 'trian_loss_u')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[1, 2], 'valid_loss_u')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[0, 3], 'trian_loss_v')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_total[0])), np.array(trian_floss_total)[1, 3], 'valid_loss_v')
            # fig.suptitle('field loss')
            # fig.savefig(os.path.join(work_path, 'field_loss_abs.svg'))
            # plt.close(fig)
            #
            # fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=3)
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[0, 0],
            #                  'trian_loss_p')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[1, 0],
            #                  'valid_loss_p')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[0, 1],
            #                  'train_loss_t')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[1, 1],
            #                  'valid_loss_t')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[0, 2],
            #                  'trian_loss_u')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[1, 2],
            #                  'valid_loss_u')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[0, 3],
            #                  'trian_loss_v')
            # Visual.plot_loss(fig, axs, np.arange(len(log_floss_totalm[0])), np.array(train_floss_totalm)[1, 3],
            #                  'valid_loss_v')
            # fig.suptitle('field loss')
            # fig.savefig(os.path.join(work_path, 'field_loss_rel.svg'))
            # plt.close(fig)


        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 250 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_coord1, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
            # train_grid, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            # valid_grid, valid_coord,  valid_true, valid_pred = inference(valid_loader, Net_model, Device)
            valid_coord1, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            # Error_func.p = 1
            # Err1a=Error_func.abs(valid_pred,valid_true)    #(*,4) ptuv
            # Err1r=Error_func.rel(valid_pred,valid_true)
            #
            # Error_func.p=2
            # Err2a=Error_func.abs(valid_pred, valid_true)   #
            # Err2r=Error_func.rel(valid_pred, valid_true)
            #
            # Err1r_flat=np.ravel(Err1r)
            #
            # # Err = [[], []]
            # # Err[0].append(Err1r)
            # # Err[1].append(Err2r)
            #
            # fig, axs = plt.subplots(1,2, figsize=(10,10), num=2)
            # # Visual.plot_box(fig, axs[0], Err1r, legends=Visual.field_name)
            # # Visual.plot_box(fig, axs[1], Err2r, legends=Visual.field_name)
            # # fig.savefig(os.path.join(work_path, 'valid_box' + str(fig_id) + '.jpg'), dpi=600,
            # #             bbox_inches='tight')
            # # plt.close(fig)
            # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), Err1r_flat, 'train_step')
            # #Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), range(len(Err2r[0, :])), 'valid_step')
            # fig.suptitle('training loss')
            # fig.savefig(os.path.join(work_path, 'field_loss.svg'))
            # plt.close(fig)

            #
            train_coord_ne = x_normalizer.back(train_coord1)
            valid_coord_ne = x_normalizer.back(valid_coord1)
            train_coord = train_coord_ne[:, :, :, -2:]
            valid_coord = valid_coord_ne[:, :, :, -2:]

            ##shape(6773,792,40,4)   batchsize,shape[1],shape[2],

            train_true, valid_true = y_normalizer.back(train_true), y_normalizer.back(valid_true)
            train_pred, valid_pred = y_normalizer.back(train_pred), y_normalizer.back(valid_pred)

            # train_pred_new = train_pred.reshape([train_pred.shape[0], 396, 40, 4])
            # valid_pred_new = valid_pred.reshape([train_pred.shape[0], 396, 40, 4])
            # train_pred_new=np.array(train_pred_new)
            # valid_pred_new=np.array(valid_pred_new)
            # mdit={'train_pred':train_pred_new}
            # mdit1={'valid_pred':valid_pred_new}
            # scipy.io.savemat(os.path.join(work_path, 'train_pred.mat'), train_pred_new)
            # scipy.io.savemat(os.path.join(work_path, 'valid_pred.mat'), valid_pred_new)

            np.save(os.path.join(work_path, "train_true.npy"), train_true)
            np.save(os.path.join(work_path, "valid_true.npy"), valid_true)
            np.save(os.path.join(work_path, "train_pred.npy"), train_pred)
            np.save(os.path.join(work_path, "valid_pred.npy"), valid_pred)
            np.save(os.path.join(work_path, "train_coord.npy"), train_coord)
            np.save(os.path.join(work_path, "valid_coord.npy"), valid_coord)

            # x = fields.shape  # 原始场shape(6773,792,40,4)
            # batch_size, coords_x, coords_y, channel = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

            # --------------------------------保存excel
            # import pandas as pd  #train_true(12,264,40,4)  true_true_f (12*264*40*4)   train_coord (12,264,40,2)
            # train_true_f=np.array(train_true).reshape(-1,1)
            # train_pred_f = np.array(train_pred).reshape(-1, 1)
            # train_coord_f=np.array(train_coord).reshape(-1,1)
            # df_train_true = pd.DataFrame(train_true_f,columns=['true'])
            # df_train_pred = pd.DataFrame(train_pred_f,columns=['pred'])
            # df_train_coord = pd.DataFrame(train_coord_f, columns=['coord'])
            # # excel_file_name = "train_data.xlsx"
            # # writer = pd.ExcelWriter(excel_file_name, engine='xlsxwriter')
            # writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work\UNet\train.xlsx')
            # df_train_true.to_excel(writer, sheet_name='Train_True', index=False)
            # df_train_pred.to_excel(writer, sheet_name='Train_Pred', index=False)
            # df_train_coord.to_excel(writer, sheet_name='Train_coord', index=False)
            # #
            # # df.to_excel(writer, index=False)
            # writer.save()
            # _--------------------------------保存Mat
            # train_true=train_true.reshape([train_true.shape[0], , , out_dim])
            # train_pred = train_pred.reshape([train_pred.shape[0],,, out_dim])
            # valid_true = valid_true.reshape([valid_true.shape[0],,, out_dim])
            # valid_pred=valid_pred.reshape([valid_pred.shape[0], , , out_dim])
            # train_true_path='train_true.mat'
            # train_pred_path='train_pred.mat'
            # valid_true_path='valid_true.mat'
            # valid_pred_path='valid_pred.mat'
            # h5py.File(train_true_path,{'train_true':train_true})
            # h5py.File(train_pred_path, {'train_pred': train_pred})
            # h5py.File(valid_true_path, {'valid_true': valid_true})
            # h5py.File(valid_pred_path, {'valid_pred': valid_pred})
            # 除了第一个whole，其余坐标重叠
            for fig_id in range(10):
                fig, axs = plt.subplots(4, 3, figsize=(18, 8), num=4)
                axs_flat = axs.flatten()
                for ax in axs_flat:
                    ax.axis('off')
                    ax.set_frame_on(False)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id],
                                      train_coord[fig_id])  # ,,cmin_max=[[0,0], [1, 1]]
                fig.savefig(os.path.join(work_path, 'train_solution_whole' + str(fig_id) + '.jpg'), dpi=600,
                            bbox_inches='tight')
                plt.close(fig)

                # fig, axs = plt.subplots(4, 3, figsize=(18, 10), num=5)
                # axs_flat = axs.flatten()
                # for ax in axs_flat:
                #     ax.axis('off')
                #     ax.set_frame_on(False)
                # Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id],
                #                       cmin_max=[[0.0015, 0.00012], [0.0025, 0.00040]])
                # fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '_local.jpg'), dpi=600,
                #             bbox_inches='tight')
                # plt.close(fig)

                fig, axs = plt.subplots(4, 3, figsize=(18, 10), num=5)
                axs_flat = axs.flatten()
                for ax in axs_flat:
                    ax.axis('off')
                    ax.set_frame_on(False)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id],
                                      cmin_max=[[0.0010, 0.00008], [0.0025, 0.00042]])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '_local_01.jpg'), dpi=600,
                            bbox_inches='tight')
                plt.close(fig)
            # ------------保存数据

            for fig_id in range(20):
                fig, axs = plt.subplots(4, 3, figsize=(18, 8), num=6)
                axs_flat = axs.flatten()
                for ax in axs_flat:
                    ax.axis('off')
                    ax.set_frame_on(False)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_whole' + str(fig_id) + '.jpg'), dpi=600,
                            bbox_inches='tight')
                plt.close(fig)

                # fig, axs = plt.subplots(4, 3, figsize=(20, 10), num=8)
                # axs_flat = axs.flatten()
                # for ax in axs_flat:
                #     ax.axis('off')
                #     ax.set_frame_on(False)
                # Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id],
                #                       cmin_max=[[0.0010, 0.00012], [0.0025, 0.00045]])
                # fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '_local.jpg'), dpi=600,
                #             bbox_inches='tight')
                # plt.close(fig)

                fig, axs = plt.subplots(4, 3, figsize=(18, 8), num=7)
                axs_flat = axs.flatten()
                for ax in axs_flat:
                    ax.axis('off')
                    ax.set_frame_on(False)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id],
                                      cmin_max=[[0.0010, 0.00008], [0.0025, 0.00042]])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '_local_01.jpg'), dpi=600,
                            bbox_inches='tight')
                plt.close(fig)

    # #------------------------------参数记录
