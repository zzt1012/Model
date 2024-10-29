##### In the name of God #####
##### Prediction of Fluid Flow in Porous Media by Sparse Observations and Physics-Informed PointNet #####

# Author: Ali Kashefi (kashefi@stanford.edu)
# Description: Implementation of Physics-informed PointNet for a 2D Porous Medium using Weakly Supervised Learning
# Version: 1.0

##### Citation #####
# If you use the code, please cite the following journal papers:
# 1. Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries
# https://doi.org/10.1016/j.jcp.2022.111510

# @article{Kashefi2022PIPN,
# title = {Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries}, <br>
# journal = {Journal of Computational Physics},
# volume = {468},
# pages = {111510},
# year = {2022},
# issn = {0021-9991},
# author = {Ali Kashefi and Tapan Mukerji}}

# 2. Prediction of fluid flow in porous media by sparse observations and physics-informed PointNet
# https://doi.org/10.1016/j.neunet.2023.08.006

# @article{kashefi2023PorousMediaPIPN,
# title={Prediction of fluid flow in porous media by sparse observations and physics-informed PointNet},
# author={Kashefi, Ali and Mukerji, Tapan},
# journal={Neural Networks},
# year={2023},
# publisher={Elsevier}}

##### Importing libraries #####
# As a first step, we import the necessary libraries.
# We use [Matplotlib](https://matplotlib.org/) for visualization purposes and [NumPy](https://numpy.org/) for computing on arrays.


import numpy as np
import matplotlib
from torch.cuda import device
from torch.profiler import schedule

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = '12'


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import grad

# Global variables
data = 1  # number of domains
Nd = 2  # dimension of problems (x,y)
N_boundary = 1  # number of points on the boundary (will be adapted later)
num_points = 1  # number of total points (will be adapted later)
category = 3  # number of variables, i.e., velocity in the x direction, velocity in the y direction, and pressure
full_list = []  # nodes in the whole domain
BC_list = []  # nodes located on the boundary
interior_list = []  # nodes inside the domain bun not on the boundary

# Training parameters
J_Loss = 0.0025  # Loss criterion
LR = 0.0003  # learning rate
Np = 250000  # Number of epochs
Nb = 1  # batch size, note: Nb should be less than data
Ns = 2.0  # scaling the network
pointer = np.zeros(shape=[Nb], dtype=int)  # to save indices of batch numbers


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Functions
def mat_mul(AA, BB):
    return torch.matmul(AA, BB)


def exp_dim(global_feature, num_points):
    return torch.tile(global_feature, [1, num_points, 1])


def compute_u(Y):
    return Y[0][:, :, 0]


def compute_v(Y):
    return Y[0][:, :, 1]


def compute_p(Y):
    return Y[0][:, :, 2]


def compute_dp_dx(X, Y):
    return grad(Y[0][:, :, 2], X)[0][:, :, 0]


def compute_dp_dy(X, Y):
    return grad(Y[0][:, :, 2], X)[0][:, :, 1]


# plot loss
def plotCost(Y, name, title):
    plt.plot(Y)
    plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(name + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(name + '.eps', bbox_inches='tight')
    plt.clf()
    # plt.show()


# plotting geometry
def plotGeometry2DPointCloud(X, name, i):
    x_p = X[i, :, 0]
    y_p = X[i, :, 1]
    plt.scatter(x_p, y_p)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name + '.png', dpi=300)
    # plt.savefig(name+'.eps')
    plt.clf()
    # plt.show()


# plotting point cloud solutions
def plotSolutions2DPointCloud(S, index, title, flag, name):
    U = np.zeros(num_points, dtype=float)
    if flag == False:
        for i in range(num_points):
            U[i] = S[index][i]
    if flag == True:
        U = S
    x_p = X_train[index, :, 0]
    y_p = X_train[index, :, 1]
    marker_size = 1.0
    plt.scatter(x_p / 10.0, y_p / 10.0, marker_size, U, cmap='jet')
    cbar = plt.colorbar()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name + '.png', dpi=300)
    plt.savefig(name + '.eps')
    plt.clf()
    # plt.show()


# plotting errors
def plotErrors2DPointCloud(Uexact, Upredict, index, title, name):
    Up = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i]

    x_p = X_train[index, :, 0]
    y_p = X_train[index, :, 1]
    marker_size = 1.0
    plt.scatter(x_p / 10.0, y_p / 10.0, marker_size, np.absolute(Uexact - Up), cmap='jet')
    cbar = plt.colorbar()
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(name + '.png', dpi=300)
    plt.savefig(name + '.eps')
    plt.clf()
    # plt.show()


# Compute L2 error
def computeRelativeL2(Uexact, Upredict, index):
    Up = np.zeros(num_points, dtype=float)
    for i in range(num_points):
        Up[i] = Upredict[index][i]

    sum1 = 0
    sum2 = 0
    for i in range(num_points):
        sum1 += np.square(Up[i] - Uexact[i])
        sum2 += np.square(Uexact[i])

    return np.sqrt(sum1 / sum2)


# Reading Data
# for the spatial correlation length (l_c) of 1.7 mm, num_gross = 4231
# for the spatial correlation length (l_c) of 0.9 mm, num_gross = 8727
# for the spatial correlation length (l_c) of 0.5 mm, num_gross = 17661

num_gross = 4231
X_train = np.random.normal(size=(data, num_points, Nd))
CFD_train = np.random.normal(size=(data, num_points, category))
X_train_mini = np.random.normal(size=(Nb, num_points, Nd))

x_pore = np.zeros(shape=(num_gross), dtype=float)
y_pore = np.zeros(shape=(num_gross), dtype=float)
u_pore = np.zeros(shape=(num_gross), dtype=float)
v_pore = np.zeros(shape=(num_gross), dtype=float)
p_pore = np.zeros(shape=(num_gross), dtype=float)


# Reading the data
def readPorous():
    coord = 0
    with open('data/Spatial_Correlation_Length_17/u.txt', 'r') as f:
        for line in f:
            x_pore[coord] = float(line.split()[0]) * 0.001 / 0.001
            y_pore[coord] = float(line.split()[1]) * 0.001 / 0.001
            u_pore[coord] = float(line.split()[2]) / 0.001
            coord += 1
    f.close()

    coord = 0
    with open('data/Spatial_Correlation_Length_17/v.txt', 'r') as f:
        for line in f:
            v_pore[coord] = float(line.split()[2]) / 0.001
            coord += 1
    f.close()

    coord = 0
    with open('data/Spatial_Correlation_Length_17/p.txt', 'r') as f:
        for line in f:
            p_pore[coord] = float(line.split()[2]) / 00.01
            coord += 1
    f.close()


readPorous()

# Boundary points  筛选出边界点的坐标（速度u、v接近0,因为设置了无滑移边界条件）
car_bound = 0
for i in range(len(x_pore)):
    if (np.absolute(u_pore[i]) < np.power(10, -22.0) and np.absolute(v_pore[i]) < np.power(10, -22.0)):
        car_bound += 1  # 计算流速接近0的边界点个数

x_bound = np.zeros(shape=(car_bound), dtype=float)
y_bound = np.zeros(shape=(car_bound), dtype=float)
index_bound = np.zeros(shape=(car_bound), dtype=int)

car_bound = 0
for i in range(len(x_pore)):
    if (np.absolute(u_pore[i]) < np.power(10, -22.0) and np.absolute(v_pore[i]) < np.power(10, -22.0)):
        x_bound[car_bound] = x_pore[i]
        y_bound[car_bound] = y_pore[i]
        index_bound[car_bound] = i
        car_bound += 1

N_boundary = car_bound
# for the spatial correlation length (l_c) of 1.7 mm, num_points = 4231
# for the spatial correlation length (l_c) of 0.9 mm, num_points = 8727
# for the spatial correlation length (l_c) of 0.5 mm, num_points = 17661

num_points = 4231  # memory sensetive

interior_point = num_points - N_boundary
X_train = np.random.normal(size=(data, num_points, Nd))  # （1,4231,2）
CFD_train = np.random.normal(size=(data, num_points, category))  # （1,4231,3）
X_train_mini = np.random.normal(size=(Nb, num_points, Nd))  # （1,4231,2）

# 填充边界点的坐标及对应数据
for i in range(data):
    for k in range(N_boundary):
        X_train[i][k][0] = x_pore[index_bound[k]]
        X_train[i][k][1] = y_pore[index_bound[k]]
        CFD_train[i][k][0] = u_pore[index_bound[k]]
        CFD_train[i][k][1] = v_pore[index_bound[k]]
        CFD_train[i][k][2] = p_pore[index_bound[k]]

    index_rest = np.arange(num_points)  # 创建所有点的索引
    index_rest[~np.isin(index_rest, index_bound)]  # 去掉边界点的索引
    print(len(index_rest))

    # 填充内部点的坐标及对应数据
    for k in range(N_boundary, num_points):
        X_train[i][k][0] = x_pore[index_rest[k - N_boundary]]
        X_train[i][k][1] = y_pore[index_rest[k - N_boundary]]
        CFD_train[i][k][0] = u_pore[index_rest[k - N_boundary]]
        CFD_train[i][k][1] = v_pore[index_rest[k - N_boundary]]
        CFD_train[i][k][2] = p_pore[index_rest[k - N_boundary]]

# Sparse Observations
# for the spatial correlation length (l_c) of 1.7 mm, k_c = 27
# for the spatial correlation length (l_c) of 0.9 mm, k_c = 27
# for the spatial correlation length (l_c) of 0.5 mm, k_c = 40


# 稀疏观测点的坐标
k_c = 27
counting = 0
x_pre_sparse = np.random.normal(size=(data, k_c * k_c))
y_pre_sparse = np.random.normal(size=(data, k_c * k_c))
for k in range(data):
    for i in range(k_c):
        for j in range(k_c):
            x_pre_sparse[k][counting] = 100 * (i * (0.64 / k_c) + 0.01)
            y_pre_sparse[k][counting] = 100 * (j * (0.64 / k_c) + 0.01)
            counting += 1

# 稀疏观测点的索引，计算每个内部点到稀疏观测点的距离，如果距离小于2.0，则将该内部点作为稀疏观测点
set_point = []
for i in range(k_c * k_c):
    x_i = x_pre_sparse[0][i]
    y_i = y_pre_sparse[0][i]
    di = np.random.normal(size=(num_points - N_boundary, 2))
    for index in range(N_boundary, num_points):
        di[index - N_boundary][0] = 1.0 * index
        di[index - N_boundary][1] = np.sqrt(
            np.power(X_train[0][index][0] - x_i, 2.0) + np.power(X_train[0][index][1] - y_i, 2.0))
    di = di[np.argsort(di[:, 1])]
    if di[0][1] < 2.0:
        set_point.append(int(di[0][0]))

sparse_n = len(set_point)
sparse_list = [[-1 for i in range(sparse_n)] for j in range(data)]

print('number of sensors:')
print(sparse_n)


# Problem set
def problemSet():
    for i in range(N_boundary):
        BC_list.append(i)

    for i in range(num_points):
        full_list.append(i)

    for i in range(data):
        for j in range(sparse_n):
            sparse_list[i][j] = set_point[j]

    for i in range(num_points):
        if i in BC_list:
            continue
        interior_list.append(i)


problemSet()

# 创建数组，存储稀疏观测点的坐标及对应数据
u_sparse = np.random.normal(size=(data, sparse_n))
v_sparse = np.random.normal(size=(data, sparse_n))
p_sparse = np.random.normal(size=(data, sparse_n))
x_sparse = np.random.normal(size=(data, sparse_n))
y_sparse = np.random.normal(size=(data, sparse_n))

for i in range(data):
    for k in range(sparse_n):
        u_sparse[i][k] = CFD_train[i][sparse_list[i][k]][0]
        v_sparse[i][k] = CFD_train[i][sparse_list[i][k]][1]
        p_sparse[i][k] = CFD_train[i][sparse_list[i][k]][2]

        x_sparse[i][k] = X_train[i][sparse_list[i][k]][0]
        y_sparse[i][k] = X_train[i][sparse_list[i][k]][1]

# Plot sparse points
plt.scatter(x_bound / 10, y_bound / 10, s=1.0)
plt.scatter(x_sparse[0, :] / 10, y_sparse[0, :] / 10, s=1.0)
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('sparse.png', dpi=300)
plt.savefig('sparse.eps')
plt.clf()

viscosity = 0.001 / 0.1  # Pa.s
density = 1.0  # kg/m^3

cfd_u = np.zeros(data * num_points)
cfd_v = np.zeros(data * num_points)
cfd_p = np.zeros(data * num_points)

counter = 0
for j in range(data):
    for i in range(num_points):
        cfd_u[counter] = CFD_train[j][i][0]
        cfd_v[counter] = CFD_train[j][i][1]
        cfd_p[counter] = CFD_train[j][i][2]
        counter += 1


def CFDsolution_u(index):
    return CFD_train[index, :, 0]


def CFDsolution_v(index):
    return CFD_train[index, :, 1]


def CFDsolution_p(index):
    return CFD_train[index, :, 2]


# Physics-informed PointNet
class PointNet(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNet, self).__init__()
        self.global_feat = global_feat
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        num_points = x.size()[1]
        x = x.permute(0, 2, 1)
        x = F.tanh(self.bn1(self.conv1(x)))
        local_x = x  # 1*64*4231

        x = F.tanh(self.bn2(self.conv2(x)))  # 1*128*4231
        x = F.tanh(self.bn3(self.conv3(x)))  # 1*1024*4231

        # x = torch.max(x, 2, keepdim=True)[0]  #1*1024*1
        # x = x.view(x.size(0), -1)   #1*1024

        x = torch.concatenate([x, local_x], 1)  # 1*1088*4231

        x = F.tanh(self.bn4(self.conv4(x)))
        x = F.tanh(self.bn5(self.conv5(x)))
        x = F.tanh(self.bn6(self.conv6(x)))
        # x = x.view(-1, 128, 1).repeat(1, 1, num_points)
        x = self.conv7(x)
        x = x.permute(0, 2, 1)

        return x


# Example usage
num_points = 4231  # Number of points in the input
Nd = 2  # Number of features for each point
Ns = 2  # Scaling factor for number of features
category = 3  # Number of output categories

# Example input
model = PointNet(global_feat=True)
model.to(device)

X = torch.rand(1, num_points, Nd)
X = X.to(device)
print(X.device)

output = model(X)# （1, 4231, 3）
output.to(device)
print('output_shape:', output.shape)  # Expected output shape: (1, category, num_points)

cost_BC = torch.tensor(0.0, dtype=torch.long)
cost_sparse = torch.tensor(0.0, dtype=torch.long)
cost_interior = torch.tensor(0.0, dtype=torch.long)

pose_BC = torch.tensor([], dtype=torch.long)
pose_sparse = torch.tensor([], dtype=torch.long)
pose_interior = torch.tensor([], dtype=torch.long)

pose_BC_p = torch.tensor([], dtype=torch.long)
pose_sparse_p = torch.tensor([], dtype=torch.long)
pose_interior_p = torch.tensor([], dtype=torch.long)


# Loss function
def ComputeCost_SE(device, X, model, viscosity, cfd_u, cfd_v, cfd_p, pose_BC, pose_sparse, pose_interior, pose_BC_p,
                   pose_sparse_p, pose_interior_p):
    # Ensure gradients are being calculated
    X.requires_grad = True
    X = X.to(device)
    Y = model(X)

    # Calculate gradients using autograd
    du_dx_in = grad(Y[0][:, 0], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 0]))[0][..., 0].view(-1)[
        pose_interior_p]
    du_dy_in = grad(Y[0][:, 0], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 0]))[0][..., 1].view(-1)[
        pose_interior_p]
    dv_dx_in = grad(Y[0][:, 1], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 1]))[0][..., 0].view(-1)[
        pose_interior_p]
    dv_dy_in = grad(Y[0][:, 1], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 1]))[0][..., 1].view(-1)[
        pose_interior_p]
    dp_dx_in = grad(Y[0][:, 2], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 2]))[0][..., 0].view(-1)[
        pose_interior_p]
    dp_dy_in = grad(Y[0][:, 2], X, create_graph=True, grad_outputs=torch.ones_like(Y[0][:, 2]))[0][..., 1].view(-1)[
        pose_interior_p]

    d2u_dx2_in = grad(du_dx_in, X, grad_outputs=torch.ones_like(du_dx_in), create_graph=True)[0][..., 0].view(-1)[
        pose_interior_p]
    d2u_dy2_in = grad(du_dy_in, X, grad_outputs=torch.ones_like(du_dy_in), create_graph=True)[0][..., 1].view(-1)[
        pose_interior_p]
    d2v_dx2_in = grad(dv_dx_in, X, grad_outputs=torch.ones_like(du_dx_in), create_graph=True)[0][..., 0].view(-1)[
        pose_interior_p]
    d2v_dy2_in = grad(dv_dy_in, X, grad_outputs=torch.ones_like(dv_dy_in), create_graph=True)[0][..., 1].view(-1)[
        pose_interior_p]

    # Residuals
    r1 = dp_dx_in - viscosity * (d2u_dx2_in + d2u_dy2_in)
    r2 = dp_dy_in - viscosity * (d2v_dx2_in + d2v_dy2_in)
    r3 = du_dx_in + dv_dy_in

    # Boundary and sparse values
    u_boundary = Y[0][:, 0].view(-1)[pose_BC_p]
    u_sparse = Y[0][:, 0].view(-1)[pose_sparse_p]
    v_boundary = Y[0][:, 1].view(-1)[pose_BC_p]
    v_sparse = Y[0][:, 1].view(-1)[pose_sparse_p]
    p_sparse = Y[0][:, 2].view(-1)[pose_sparse_p]

    # Ground truth values
    # sparse_u_truth = cfd_u[pose_sparse]
    # sparse_v_truth = cfd_v[pose_sparse]
    # sparse_p_truth = cfd_p[pose_sparse]

    sparse_u_truth = torch.tensor(cfd_u[pose_sparse], dtype=u_sparse.dtype, device=u_sparse.device)
    sparse_v_truth = torch.tensor(cfd_v[pose_sparse], dtype=v_sparse.dtype, device=v_sparse.device)
    sparse_p_truth = torch.tensor(cfd_p[pose_sparse], dtype=p_sparse.dtype, device=p_sparse.device)

    # Costs
    PDE_cost = torch.mean(r1 ** 2 + r2 ** 2 + r3 ** 2)
    BC_cost = torch.mean((u_boundary - 0.0) ** 2 + (v_boundary - 0.0) ** 2)

    Sparse_cost = torch.mean(
        (u_sparse - sparse_u_truth) ** 2 + (v_sparse - sparse_v_truth) ** 2 + (p_sparse - sparse_p_truth) ** 2)
    Sparse_cost = Sparse_cost.detach().cpu().numpy()

    return (100.0 * PDE_cost + 100.0 * Sparse_cost + BC_cost)


# 训练
def build_model_PIPN_PorousMedia(device, model, X_train, BC_list, sparse_list, interior_list, num_points, data, Np, Nb, LR,
                                 J_Loss):
    min_loss = 1000
    converge_iteration = 0
    criteria = J_Loss

    cost_all = []

    u_final = np.zeros((data, num_points), dtype=float)
    v_final = np.zeros((data, num_points), dtype=float)
    p_final = np.zeros((data, num_points), dtype=float)
    dp_dx_final = np.zeros((data, num_points), dtype=float)
    dp_dy_final = np.zeros((data, num_points), dtype=float)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Np, gamma=0.5)

    for epoch in range(Np):
        temp_cost = 0
        arr = np.arange(data)
        np.random.shuffle(arr)

        for sb in range(int(data / Nb)):
            pointer = arr[int(sb * Nb):int((sb + 1) * Nb)]

            # Prepare group_BC, group_sparse, and group_interior 索引数组
            group_BC = np.zeros(int(len(pointer) * len(BC_list)), dtype=int)
            group_sparse = np.zeros(int(len(pointer) * len(sparse_list[0])), dtype=int)
            group_interior = np.zeros(int(len(pointer) * len(interior_list)), dtype=int)

            # Prepare group_BC， catch 是索引
            catch = 0
            for ii in range(len(pointer)):
                for jj in range(len(BC_list)):
                    group_BC[catch] = int(pointer[ii] * num_points + jj)
                    catch += 1

            # Prepare group_sparse
            catch = 0
            for ii in range(len(pointer)):
                for jj in range(len(sparse_list[0])):
                    group_sparse[catch] = sparse_list[pointer[ii]][jj] + pointer[ii] * num_points
                    catch += 1

            # Prepare group_interior
            catch = 0
            for ii in range(len(pointer)):
                for jj in range(len(interior_list)):
                    group_interior[catch] = int(pointer[ii] * num_points + len(BC_list) + jj)
                    catch += 1

            # Prepare pose_BC, pose_sparse, and pose_interior
            group_BC_p = np.zeros(int(len(pointer) * len(BC_list)), dtype=int)
            group_sparse_p = np.zeros(int(len(pointer) * sparse_n), dtype=int)
            group_interior_p = np.zeros(int(len(pointer) * len(interior_list)), dtype=int)

            catch = 0
            for ii in range(Nb):
                for jj in range(len(BC_list)):
                    group_BC_p[catch] = int(ii * num_points + jj)
                    catch += 1

            catch = 0
            for ii in range(Nb):
                for jj in range(sparse_n):
                    group_sparse_p[catch] = sparse_list[pointer[ii]][jj] + ii * num_points
                    catch += 1

            catch = 0
            for ii in range(Nb):
                for jj in range(len(interior_list)):
                    group_interior_p[catch] = int(ii * num_points + len(BC_list) + jj)
                    catch += 1

            pose_BC = group_BC
            pose_sparse = group_sparse
            pose_interior = group_interior

            pose_BC_p = group_BC_p
            pose_sparse_p = group_sparse_p
            pose_interior_p = group_interior_p

            X_train_mini = np.take(X_train, pointer[:], axis=0)
            X_train_mini = torch.tensor(X_train_mini, dtype=torch.float32)  # （1,4231,2）
            X_train_mini = X_train_mini.to(device)

            outputs = model(X_train_mini) # （1,4231,3）

            # Forward pass and compute
            optimizer.zero_grad()
            # Compute costs (assuming ComputeCost_SE and compute_* functions are adapted to PyTorch)
            cost = ComputeCost_SE(device, X, model, viscosity, cfd_u, cfd_v, cfd_p, pose_BC, pose_sparse, pose_interior, pose_BC_p, pose_sparse_p, pose_interior_p)
            temp_cost += cost.item()

            # Backward pass and optimization
            cost.backward()
            optimizer.step()

            cost_all.append(temp_cost)


            print("Epoch:", epoch)
            print("Temp_cost:", temp_cost)

            if epoch > 0 and epoch % 10 == 0:
                    plotCost(cost_all, 'cost', 'loss function')

            if epoch > 0 and epoch % 1000 == 0:

                # Error Analysis
                error_u_rel = []
                error_v_rel = []
                error_p_rel = []

                for index in range(data):
                    plotSolutions2DPointCloud(CFDsolution_u(index), index, 'Ground truth $\it{u}$ (mm/s)', True,
                                              'u truth')
                    plotSolutions2DPointCloud(u_final, index, 'Prediction $\it{u}$ (mm/s)', False, 'u prediction')
                    plotSolutions2DPointCloud(CFDsolution_v(index), index, 'Ground truth $\it{v}$ (mm/s)', True,
                                              'v truth')
                    plotSolutions2DPointCloud(v_final, index, 'Prediction $\it{v}$ (mm/s)', False, 'v prediction')
                    plotSolutions2DPointCloud(CFDsolution_p(index) / 10.0, index, 'Ground truth $\it{p}$ (Pa)', True,
                                              'p truth')
                    plotSolutions2DPointCloud(p_final / 10.0, index, 'Prediction $\it{p}$ (Pa)', False, 'p prediction')

                    plotErrors2DPointCloud(CFDsolution_u(index), u_final, index,
                                           'Absolute error ' + '$\it{u}$' + ' (mm/s)',
                                           'u error')
                    plotErrors2DPointCloud(CFDsolution_v(index), v_final, index,
                                           'Absolute error ' + '$\it{v}$' + ' (mm/s)',
                                           'v error')
                    plotErrors2DPointCloud(CFDsolution_p(index) / 10.0, p_final / 10.0, index,
                                           'Absolute error ' + '$\it{p}$' + ' (Pa)', 'p error')


                    error_u_rel.append(computeRelativeL2(CFDsolution_u(index), u_final, index))
                    error_v_rel.append(computeRelativeL2(CFDsolution_v(index), v_final, index))
                    error_p_rel.append(computeRelativeL2(CFDsolution_p(index), p_final, index))

                for index in range(data):
                    print('\n')
                    print(index)
                    print('error_u_rel:')
                    print(error_u_rel[index])
                    print('error_v_rel:')
                    print(error_v_rel[index])
                    print('error_p_rel:')
                    print(error_p_rel[index])
                    print('\n')




            if temp_cost < min_loss:
                u_final = np.power(outputs[:, :, 0].detach().cpu().numpy(), 1.0)
                v_final = np.power(outputs[:, :, 1].detach().cpu().numpy(), 1.0)
                p_final = np.power(outputs[:, :, 2].detach().cpu().numpy(), 1.0)

            # Update min_loss
                min_loss = temp_cost
            converge_iteration = epoch

            if min_loss < criteria:
                break

build_model_PIPN_PorousMedia(device, model, X_train, BC_list, sparse_list, interior_list, num_points, data, Np, Nb, LR, J_Loss)



