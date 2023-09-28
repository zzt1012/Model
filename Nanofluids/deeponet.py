import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
from torch.utils.data import DataLoader
from utilize.process_data import DataNormer, MatLoader
from don.DeepONets import DeepONetMulti
from utilize.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import sys
from run_MLP import get_grid, get_origin
from torchinfo import summary
from utilize.loss_metrics import FieldsLpLoss

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
    train_floss_u = 0
    train_floss_v = 0
    train_floss_total = 0

    train_floss_pm = 0
    train_floss_tm = 0
    train_floss_um = 0
    train_floss_vm = 0
    train_floss_totalm = 0

    train_loss = 0
    for batch, (f, x, u) in enumerate(dataloader):
        f = f.to(device)   #（1,792,40,12）
        x = x.to(device)   #（1,6773*792*40,2）
       # x1=x[:,:,-2:]
        u = u.to(device)  #（1,25344，4）
        pred = netmodel([f, ], x, size_set=False)    #f desigin 12   x coords 4    x1在forward只取了后三维

        loss = lossfunc(pred, u)

        field_floss_p = fieldloss.abs(pred, u)[:, 0].mean()  # rel/abs计算出（32,4）
        field_floss_t = fieldloss.abs(pred, u)[:, 1].mean()
        field_floss_u = fieldloss.abs(pred, u)[:, 2].mean()
        field_floss_v = fieldloss.abs(pred, u)[:, 3].mean()

        field_floss_pm = fieldloss.abs(pred, u)[:, 0].max()  # rel/abs计算出（32,4）
        field_floss_tm = fieldloss.abs(pred, u)[:, 1].max()
        field_floss_um = fieldloss.abs(pred, u)[:, 2].max()
        field_floss_vm = fieldloss.abs(pred, u)[:, 3].max()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_floss_p += field_floss_p.item()
        train_floss_t += field_floss_t.item()
        train_floss_u += field_floss_u.item()
        train_floss_v += field_floss_v.item()

        train_floss_pm += field_floss_pm.item()
        train_floss_tm += field_floss_tm.item()
        train_floss_um += field_floss_um.item()
        train_floss_vm += field_floss_vm.item()


    scheduler.step()
    return train_loss / (batch + 1), train_floss_p / (batch + 1) , \
           train_floss_t / (batch + 1) , train_floss_u / (batch + 1) , \
           train_floss_v / ( batch + 1) , train_floss_pm / (batch + 1) , \
           train_floss_tm / (batch + 1) , train_floss_um / (batch + 1) , \
           train_floss_vm / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc, fieldloss):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
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

    with torch.no_grad():
        for batch, (f, x, u) in enumerate(dataloader):
            f = f.to(device)
            x = x.to(device)
           # x1=x[:,:,-2:]
            u = u.to(device)
            pred = netmodel([f, ], x, size_set=False)

            loss = lossfunc(pred, u)

            field_floss_p = fieldloss.abs(pred, u)[:, 0].mean()  # rel/abs计算出（32,4）
            field_floss_t = fieldloss.abs(pred, u)[:, 1].mean()
            field_floss_u = fieldloss.abs(pred, u)[:, 2].mean()
            field_floss_v = fieldloss.abs(pred, u)[:, 3].mean()

            field_floss_pm = fieldloss.abs(pred, u)[:, 0].max()  # rel/abs计算出（32,4）
            field_floss_tm = fieldloss.abs(pred, u)[:, 1].max()
            field_floss_um = fieldloss.abs(pred, u)[:, 2].max()
            field_floss_vm = fieldloss.abs(pred, u)[:, 3].max()

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

    return valid_loss / (batch + 1), valid_floss_p / (batch+1), \
           valid_floss_t /(batch+1) , valid_floss_u/(batch+1), \
           valid_floss_v / (batch+1),valid_floss_pm / (batch+1) , \
           valid_floss_tm /(batch+1) , valid_floss_um/(batch+1), \
           valid_floss_vm / (batch+1)



def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with torch.no_grad():
        f, x, u = next(iter(dataloader))     #u真实物理场  x：coords   f:design_ 1维，    【f,]2维列表
        f = f.to(device)
        x = x.to(device)
        #x1 = x[:, :, -2:]
        pred = netmodel([f, ], x, size_set=False)
        #f1=[f,]

    # equation = model.equation(u_var, y_var, out_pred)
    return x.cpu().numpy(), x.cpu().numpy(), u.numpy(), pred.cpu().numpy()   #train_source,train_coord,此时train_coord是input后两个通道


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'deepONet'
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


    # 将控制台的结果输出到log文件
    # sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')


    in_dim = 2
    out_dim = 4

    ntrain = 4000
    nvalid = 500
    batch_size = 32
    batch_size2 = batch_size


    epochs = 1001
    learning_rate = 0.01
    scheduler_step = [800, 900, 1000]
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################

#____________________________

    file_path = os.path.join('data', 'dim_pro8_single_all.mat')
    reader = MatLoader(file_path)
    fields = reader.read_field('field')
    design = reader.read_field('data')
    coords = reader.read_field('grids')[..., :2]
    design_tile = torch.tile(design[:, None, None, :], (1, 792, 40, 1))
    input_size = design_tile.shape[1:]

    design_shape = design.shape[-1]
    fields_shape = fields.shape[-1]
    coords_shape = coords.shape[-1]
    #design_tile_s=design_tile.shape[1]   #792
#__________________________________________

   # design_=[design, ]
    output = fields
   # output = torch.tensor(output, dtype=torch.float)

   # print(design_.shape, fields.shape)

    train_f = design[:ntrain, :]      #train_f不要绑定空间坐标
    train_u = output[:ntrain,:,:,:] 
    valid_f = design[ntrain:ntrain + nvalid, :]
    valid_u = output[ntrain:ntrain + nvalid, :,:,:]
    # train_coord = torch.tile(coords, [train_f.shape[0], 1])  #所有样本的坐标是一致的。
    # valid_coord = torch.tile(coords, [valid_f.shape[0], 1])

    train_coord = coords[:ntrain,:,:,:]
    valid_coord = coords[ntrain:ntrain + nvalid, :,:,:]

    # u_show = train_u.numpy()
    # gird_show = train_grid.numpy()


#----------design_归一化
    f_normalizer = DataNormer(train_f.numpy(), method='min-max')
    train_f = f_normalizer.norm(train_f)
    valid_f = f_normalizer.norm(valid_f)
    f_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))

#-------------真实场归一化
    u_normalizer = DataNormer(train_u.numpy(), method='min-max')
    train_u = u_normalizer.norm(train_u)
    valid_u = u_normalizer.norm(valid_u)
    u_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))

    grid_normalizer = DataNormer(train_coord.numpy(), method='min-max')
    train_coord = grid_normalizer.norm(train_coord)
    valid_coord = grid_normalizer.norm(valid_coord)

    # train_coord = train_coord.reshape([train_u.shape[0], -1, 2])
    # valid_coord = valid_coord.reshape([valid_u.shape[0], -1, 2])
    # train_u = train_u.reshape([train_u.shape[0],-1, out_dim])
    # valid_u = valid_u.reshape([valid_u.shape[0],-1, out_dim])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_f, train_coord, train_u),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_f, valid_coord, valid_u),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    #算子维度是设计变量,此处把设计变量和空间坐标分开，因为可以写多个branch_net（包含设计变量或边界条件的）
    Net_model = DeepONetMulti(input_dim=2, operator_dims=[10, ], output_dim=4,
                              planes_branch=[64] * 4, planes_trunk=[64] * 3).to(Device)

    input1 = torch.randn(batch_size, train_u.shape[1], train_u.shape[2], 10).to(Device)
    input2 = torch.randn(batch_size,  train_u.shape[1],  train_u.shape[2], 2).to(Device)
    summary(Net_model, input_data=[input1,input2], device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    Error_func = FieldsLpLoss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=0)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))

    star_time = time.time()
    log_loss = [[], []]
    log_floss_p = [[], []]
    log_floss_t = [[], []]
    log_floss_u = [[], []]
    log_floss_v = [[], []]
    log_floss_pm = [[], []]
    log_floss_tm = [[], []]
    log_floss_um = [[], []]
    log_floss_vm = [[], []]

    log_floss_total = [[], []]
    log_floss_totalm = [[], []]

    ################################################################
    # train process
    ################################################################

    # 生成网格文件

    for epoch in range(epochs):

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
        valid_loss, valid_floss_p, valid_floss_t, valid_floss_u, valid_floss_v, \
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
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 1 == 0:
            import pandas as pd
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
            writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work_new_loss2\deeponet\4000_500\loss.xlsx')
            df.to_excel(writer, index=False)
            writer.save()

            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=2)
            Visual.plot_loss(fig, axs, np.arange(len(log_floss_p[0])), np.array(log_floss_p)[0, :], 'trian_loss_p')
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
            import pandas as pd

            x = np.arange(len(log_floss_p[0]))
            train_loss_p = np.array(log_floss_p)[0, :]
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
                               'train_loss_v': train_loss_v, 'valid_loss_v': valid_loss_v, })
            writer = pd.ExcelWriter(
                r'D:\pythonProject\applied_sciences\work_new_loss2\deeponet\4000_500\field_loss_mean.xlsx')
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
            writer = pd.ExcelWriter(r'D:\pythonProject\applied_sciences\work_new_loss2\deeponet\4000_500\field_loss_max.xlsx')
            df.to_excel(writer, index=False)
            writer.save()

        ################################################################
        # Visualization
        ################################################################
        if epoch > 0 and epoch % 200 == 0:

            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord= grid_normalizer.back(train_coord)
            valid_coord= grid_normalizer.back(valid_coord)
            train_true, valid_true = u_normalizer.back(train_true), u_normalizer.back(valid_true)
            train_pred, valid_pred = u_normalizer.back(train_pred), u_normalizer.back(valid_pred)

            # train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            # train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            # valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            # valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            np.save(os.path.join(work_path, "train_true.npy"), train_true)
            np.save(os.path.join(work_path, "valid_true.npy"), valid_true)
            np.save(os.path.join(work_path, "train_pred.npy"), train_pred)
            np.save(os.path.join(work_path, "valid_pred.npy"), valid_pred)
            np.save(os.path.join(work_path, "train_coord.npy"), train_coord)


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
