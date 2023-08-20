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

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    train_loss = 0
    for batch, (f, x, u) in enumerate(dataloader):
        f = f.to(device)   #（1,792,40,12）
        f1=f[:,:,:,:10]
        x = x.to(device)   #（1,6773*792*40,2）
       # x1=x[:,:,-2:]
        u = u.to(device)  #（1,25344，4）
        pred = netmodel([f1, ], x, size_set=False)    #f desigin 12   x coords 4    x1在forward只取了后三维

        loss = lossfunc(pred, u)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with torch.no_grad():
        for batch, (f, x, u) in enumerate(dataloader):
            f = f.to(device)
            f1 = f[:, :, :, :10]
            x = x.to(device)
           # x1=x[:,:,-2:]
            u = u.to(device)
            pred = netmodel([f1, ], x, size_set=False)

            loss = lossfunc(pred, u)

            valid_loss += loss.item()

    return valid_loss / (batch + 1)


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
        f1 = f[:, :, :, :10]
        x = x.to(device)
        #x1 = x[:, :, -2:]
        pred = netmodel([f1, ], x, size_set=False)
        #f1=[f,]

    # equation = model.equation(u_var, y_var, out_pred)
    return x.cpu().numpy(), x.cpu().numpy(), u.numpy(), pred.cpu().numpy()   #train_source,train_coord,此时train_coord是input后两个通道


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'deepONet'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # 将控制台的结果输出到log文件
    # sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

#    design, fields = get_origin()

    in_dim = 2
    out_dim = 4

    ntrain = 2
    nvalid = 1
    batch_size = 1
    batch_size2 = batch_size


    epochs = 401
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1



    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    # r = out_dim*64*64
    # s = 28

    ################################################################
    # load data
    ################################################################

    # grid = get_grid()
    # grid_trans = torch.tensor(grid[np.newaxis,:,:,:], dtype=torch.float)

#____________________________

    file_path = os.path.join('data', 'dim_pro8_single_all.mat')
    reader = MatLoader(file_path)
    fields = reader.read_field('field')
    # fields=fields[:,::2,:,:]  #第一、三、四个维度不变，第二个维度缩小一半
    design = reader.read_field('data')
    coords = reader.read_field('grids')[..., :2]
    design_tile = torch.tile(design[:, None, None, :], (1, 792, 40, 1))
    input_size = design_tile.shape[1:]
    # layer = UNet2d(in_sizes=input_size, out_sizes=fields.shape[1:], width=32, depth=6)

    design_shape = design.shape[-1]
    fields_shape = fields.shape[-1]
    coords_shape = coords.shape[-1]
    design_tile_s=design_tile.shape[1]   #792
#__________________________________________



    design_=[design_tile, ]
    input = torch.concat((design_tile,coords),dim=-1)
    input = torch.tensor(input, dtype=torch.float)

    output = fields
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_f = input[:ntrain, :]
    train_u = output[:ntrain,:,:,:] #这里的u还没有展开，需要先归一化再展开
    valid_f = input[ntrain:ntrain + nvalid, :]
    valid_u = output[ntrain:ntrain + nvalid, :,:,:]
    train_coord = torch.tile(coords, [train_f.shape[0], 1, 1, 1])#所有样本的坐标是一致的。
    valid_coord = torch.tile(coords, [valid_f.shape[0], 1, 1, 1])

    u_show = train_u.numpy()
    # gird_show = train_grid.numpy()


#----------design_归一化
    f_normalizer = DataNormer(train_f.numpy(), method='mean-std')
    f_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))
    train_f = f_normalizer.norm(train_f)
    valid_f = f_normalizer.norm(valid_f)

#-------------真实场归一化
    u_normalizer = DataNormer(train_u.numpy(), method='mean-std')
    u_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    train_u = u_normalizer.norm(train_u)
    valid_u = u_normalizer.norm(valid_u)

    grid_normalizer = DataNormer(train_coord.numpy(), method='mean-std')#这里的axis不一样了
    train_coord = grid_normalizer.norm(train_coord)
    valid_coord = grid_normalizer.norm(valid_coord)

    # grid_trans = grid_trans.reshape([1, -1, 2])
#
    train_coord = train_coord.reshape([train_u.shape[0], -1, 2])
    valid_coord = valid_coord.reshape([valid_u.shape[0], -1, 2])
    train_u = train_u.reshape([train_u.shape[0],-1, out_dim])
    valid_u = valid_u.reshape([valid_u.shape[0],-1, out_dim])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_f, train_coord, train_u),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_f, valid_coord, valid_u),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    #算子维度是设计变量,此处把设计变量和空间坐标分开，因为可以写多个branch_net（包含设计变量或边界条件的）
    #planes_branch是branch_net中间隐藏层，planes_trunk是trunk_net隐藏层
    Net_model = DeepONetMulti(input_dim=2, operator_dims=[10, ], output_dim=4,
                              planes_branch=[64] * 3, planes_trunk=[64] * 2).to(Device)

    # input1 = torch.randn(batch_size, train_u.shape[1], train_u.shape[2], train_u.shape[-1]).to(Device)
    # # input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
    # summary(Net_model, input_data=[input1], device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    # 生成网格文件

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 20 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################
        if epoch > 0 and epoch % 100 == 0:

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

            train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20),num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], train_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)