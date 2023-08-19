import pandas as pd
import torch
from torch.utils.data import DataLoader
#from torchinfo import summary
from FNO.FNOs import FNO2d
from utilize.process_data import DataNormer, MatLoader
from CNN.ConvNets import UNet2d, UpSampleNet2d,DownSampleNet2d
from utilize.visual_data import MatplotlibVision, TextLogger
#from FNO.FNOs import CombinedNet

import matplotlib.pyplot as plt
import time
import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn


def feature_transform(x):
    """
    Args:    x: input coordinates
    Returns:  res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1],shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args: data_loader: output fields at last time step      netmodel: Network
    """
    train_loss = 0
    batch_size=20
    batch=batch_size
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)     #xx:fields   yy：输出nu,f
        yy = yy.to(device)
        #gd = feature_transform(xx)
        pred = netmodel(xx)  #使用网络模型netmodel特征变换后的数据进行前向传播
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()  #使用优化器 optimizer的zero_grad方法将之前的梯度清零，以便进行新一轮的反向传播。
        loss.backward()  #对损失反向传播，计算梯度
        optimizer.step() #使用优化器 optimizer的step方法更新模型的参数，根据计算得到的梯度进行优化。
        train_loss += loss.item()
    scheduler.step()

        # print('训练次数'.format(train_loss, loss.item()))

    return train_loss / (batch + 1)/ batch_size

#测试步骤开始
def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:  data_loader: input coordinates    model: Network   lossfunc: Loss function
    """
    valid_loss = 0
    batch=20
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            #gd = feature_transform(xx)

            pred = netmodel(xx)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1) / batch_size


def inference(dataloader, netmodel, device):
    """
    Args:   dataloader: input coordinates   netmodel: Network
    Returns:   out_pred: predicted fields
    """
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        #gd = feature_transform(xx)
        pred = netmodel(xx)

    return xx.cpu().numpy(),  yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'Down'
    work_path = os.path.join('../work_down', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')



#_-------------------------------------first:  { ntrain:600, nvalid:100, batchsize:20, epochs:400,learning rate:0.001, scheduler_step:300, scheduler_gamma:0.1} {p小9%，f大20%}
    ####训练集参数,适度减小  (400;100)
    ntrain = 6000
    nvalid = 1000

    r1 = 1
    r2 = 3
    s1 = int(((792 - 1) / r1) + 1)
    s2 = int(((40 - 1) / r2) + 1)

    batch_size = 20
    batch_size2 = batch_size

    epochs = 501
    learning_rate = 0.001   #first:0.001
    scheduler_step = [301,401,501]
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################
    file_path = os.path.join('../data', 'dim_pro8_single_all.mat')
    reader = MatLoader(file_path)
    fields = reader.read_field('field')
    # fields=fields[:,::2,:,:]  #第一、三、四个维度不变，第二个维度缩小一半
    design = reader.read_field('data')
    target = torch.concat((reader.read_field('Nu'), reader.read_field('f')), dim=-1)
    #target=reader.read_field('Nu')
    design_tile = torch.tile(design[:, None, None, :], (1, 792, 40, 1))

    design_shape = design.shape[-1]
    fields_shape = fields.shape[-1]
    target_shape = target.shape[-1]


    input = torch.tensor(fields, dtype=torch.float)
    output = torch.tensor(target, dtype=torch.float)
   # print('in',input.shape, 'out',output.shape)
    sys.stdout.info("input sizes: {}, output sizes: {}".format(input.shape, output.shape))

    del  target

    train_x = input[:ntrain]
    train_y = output[:ntrain]
    valid_x = input[ntrain:ntrain + nvalid]
    valid_y = output[ntrain:ntrain + nvalid]
    # train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    # valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]

    #del reader
    x_normalizer = DataNormer(train_x.numpy(), method='min-max')#本设计都采用线性化归一化方式
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='min-max')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    # sys.stdout.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
    #             .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    if name == 'Down':
        Net_model = DownSampleNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[-1], width=32, depth=4).to(Device)
    elif name == 'FNO':
        Net_model = FNO2d(in_dim=train_x.shape[-1], out_dim=train_y.shape[-1], width=64, depth=4).to(Device)
#unet不涉及steps。因为只有非定常问题才会涉及steps
    elif name == 'UNet':
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:], width=32, depth=6).to(Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))

    star_time = time.time()
    log_loss = [[], []]
    pred_par = [[], []]
    ################################################################
    # train process
    ################################################################

    for epoch in range(epochs):
        #print('---第{}轮优化开始了---'.format(epoch+1))

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))
        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        #pred_par.append([Net_model.p.detach().cpu().item(), Net_model.f.detach().cpu().item()])
        pred_par[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))
        pred_par[1].append(valid(valid_loader, Net_model, Device, Loss_func))

        star_time = time.time()

        train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
        valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

        torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                   os.path.join(work_path, 'latest_model.pth'))

        # train_coord = x_normalizer.back(train_coord)
        # valid_coord = x_normalizer.back(valid_coord)
        # train_true, valid_true = y_normalizer.back(train_true), y_normalizer.back(valid_true)
        # train_pred, valid_pred = y_normalizer.back(train_pred), y_normalizer.back(valid_pred)
        #
        # data0 = []
        # for fig_id in range(epochs):  # range20
        #     pred_p = train_pred[fig_id][0]
        #     true_p = train_true[fig_id][0]
        #     error_p = abs(train_pred[fig_id][0] - train_true[-1][0]) / train_true[-1][0] * 100
        #     pred_f = train_pred[-1][1]
        #     true_f = train_true[-1][1]
        #     error_f = abs(train_pred[-1][1] - train_true[-1][1]) / train_true[-1][1] * 100
        #     row = [pred_p, true_p, error_p, pred_f, true_f, error_f]
        #     data0.append(row)
        #
        #     df = pd.DataFrame(data0, columns=['p_p', 't_p', 'error_p', 'p_f', 't_f', 'erroe_f'])
        #     df.to_excel('data0.xlsx', index=False)
        #
        #     print('pred p: {:.4f}, true p: {:.4f}, error p: {:.4f}%,'
        #           'pred f: {:.4f}, true f: {:.4f}, error f : {:.4f}%'.
        #           format(train_pred[-1][0], train_true[-1][0],
        #                  abs(train_pred[-1][0] - train_true[-1][0]) / train_true[-1][0] * 100,
        #                  train_pred[-1][1], train_true[-1][1],
        #                  abs(train_pred[-1][1] - train_true[-1][1]) / train_true[-1][1] * 100))


        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)


        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 4 == 0:

            train_coord,  train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord,  valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord = x_normalizer.back(train_coord)
            valid_coord = x_normalizer.back(valid_coord)
            train_true, valid_true = y_normalizer.back(train_true), y_normalizer.back(valid_true)
            train_pred, valid_pred = y_normalizer.back(train_pred), y_normalizer.back(valid_pred)


            #data0=[]
            # for fig_id in range(10):
            #     df = pd.DataFrame({
            #         'pred p': [train_pred[fig_id][0]],
            #         'true p': [train_true[fig_id][0]],
            #         'error_p': [abs(train_pred[fig_id][0] - train_true[fig_id][0]) / train_true[fig_id][0] * 100],
            #         'pred f': [train_pred[fig_id][1]],
            #         'true f': [train_true[fig_id][1]],
            #         'error_f': [abs(train_pred[fig_id][1] - train_true[fig_id][1]) / train_true[fig_id][1] * 100]
            #     })
            #     writer = pd.ExcelWriter(r"D:\pythonProject\applied_sciences\work_down\Down\downstream.xlsx")
            #     df.to_excel(writer, index=False)
            #     writer.save()
            try:
                dff=pd.read_excel('data0.xlsx')
            except FileNotFoundError:
                dff=pd.DataFrame()

            for fig_id in range(20):           #batchsize20
                        pred_p= train_pred[fig_id][0]   #train_pred (20,2)
                        true_p=train_true[fig_id][0]
                        error_p=abs(train_pred[fig_id][0] - train_true[fig_id][0]) / train_true[fig_id][0] * 100
                        pred_f=train_pred[fig_id][1]
                        true_f=train_true[fig_id][1]
                        error_f=abs(train_pred[fig_id][1] - train_true[fig_id][1]) / train_true[fig_id][1] * 100
                        #row=[pred_p,true_p,error_p,pred_f,true_f,error_f]
                        #data0.append(row)
                        print('pred nu: {:.4f}, true nu: {:.4f}, error nu: {:.4f}%,'
                              'pred f: {:.4f}, true f: {:.4f}, error f : {:.4f}%'.
                              format(train_pred[fig_id][0], train_true[fig_id][0],
                                     abs(train_pred[fig_id][0] - train_true[fig_id][0]) / train_true[fig_id][0] * 100,
                                     train_pred[fig_id][1], train_true[fig_id][1],
                                     abs(train_pred[fig_id][1] - train_true[fig_id][1]) / train_true[fig_id][1] * 100))
                        data0=[pred_p,true_p,error_p,pred_f,true_f,error_f]

#循环只保留了最后一个pred

                        #df1=pd.DataFrame()
                        df=pd.DataFrame([data0],columns=['p_nu','t_nu','error_nu','p_f','t_f','erroe_f'])
                        combined_df=pd.concat([dff,df],ignore_index=True)
            combined_df.to_excel('data0.xlsx',index=False)





#-------------------只能保存一个

                # df=pd.DataFrame()
                # df['pred p']= [train_pred[fig_id][0]]
                # df['true p']= [train_true[fig_id][0]]
                # df['error_p']= [abs(train_pred[fig_id][0] - train_true[fig_id][0]) / train_true[fig_id][0] * 100]
                # df['pred f']= [train_pred[fig_id][1]]
                # df['true f']= [train_true[fig_id][1]]
                # df['error_f']= [abs(train_pred[fig_id][1] - train_true[fig_id][1]) / train_true[fig_id][1] * 100]
                #
                # data2 = pd.ExcelWriter(r"D:\pythonProject\applied_sciences\work_down\Down\downall.xlsx")
                # df.to_excel(data2,sheet_name='pred',index=False)


                # fig, axs = plt.subplots(2, 1, figsize=(15, 8), num=2)
                # Visual.plot_value(np.arange(len(pred_par)), np.array(pred_par)[:, 0][fig_id], 'p_pred')
                # Visual.plot_value([0, len(pred_par) - 1], [Net_model.delta_p, Net_model.delta_p][fig_id], 'p_true')
                # plt.subplot(212)
                # Visual.plot_value(np.arange(len(pred_par)), np.array(pred_par)[:, -1][fig_id], 'f_pred')
                # Visual.plot_value([0, len(pred_par) - 1], [Net_model.f,Net_model.f][fig_id], 'f_true')
                # plt.savefig(os.path.join(work_path, 'pred_p_f.svg'))
