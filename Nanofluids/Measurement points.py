import os
import h5py
import numpy as np
import pandas as pd

from utilize.process_data import MatLoader

file_path = os.path.join('data', 'dim_pro8_single_all.mat')
reader = MatLoader(file_path)
fields = reader.read_field('field')   #shape(6773,792,40,4)
design = reader.read_field('data')
coords = reader.read_field('grids')   #(6773,792,40,3)

fields=fields[:,:,:,0]
fields=fields.permute(2,1,0)

coords=coords[:,:,:,0]
coords=coords.permute(2,1,0)

shape = fields.shape
print(shape)
batchsize, size_x, size_y = shape[2], shape[0], shape[1]


#--------------------------------------------------------------------------------
#模拟测点，上下壁面各取6个点，每间隔（792/6=）132个点取一个测点，总共取6个测点。每个测点取p和t。

dowm=fields[0,1::131,:]
print('下壁面',dowm)
ups=fields[-1,1::131,:]
print('上壁面',ups)

down=pd.DataFrame(dowm)
ups=pd.DataFrame(ups)


down1=down.iloc[0,:]
down2=down.iloc[1,:]
down3=down.iloc[2,:]
down4=down.iloc[3,:]
down5=down.iloc[4,:]
down6=down.iloc[5,:]


ups1=ups.iloc[0,:]
ups2=ups.iloc[1,:]
ups3=ups.iloc[2,:]
ups4=ups.iloc[3,:]
ups5=ups.iloc[4,:]
ups6=ups.iloc[5,:]


outfile='down_pressure.xlsx'
with pd.ExcelWriter(outfile) as writer:
    down1.to_excel(writer,sheet_name='下壁面1',index=False)
    down2.to_excel(writer, sheet_name='下壁面2', index=False)
    down3.to_excel(writer, sheet_name='下壁面3', index=False)
    down4.to_excel(writer, sheet_name='下壁面4', index=False)
    down5.to_excel(writer, sheet_name='下壁面5', index=False)
    down6.to_excel(writer, sheet_name='下壁面6', index=False)
print('保存了excel')

outfile='up_pressure.xlsx'
with pd.ExcelWriter(outfile) as writer:
    ups1.to_excel(writer,sheet_name='上壁面1',index=False)
    ups2.to_excel(writer, sheet_name='上壁面2', index=False)
    ups3.to_excel(writer, sheet_name='上壁面3', index=False)
    ups4.to_excel(writer, sheet_name='上壁面4', index=False)
    ups5.to_excel(writer, sheet_name='上壁面5', index=False)
    ups6.to_excel(writer, sheet_name='上壁面6', index=False)
print('保存了物理量excel')

#------------------------------------------------------------
#取上下壁面及中间的物理量沿x分布趋势

d1=fields[1,:,::batchsize]          #距离下壁面1e-6
print('下壁面',d1)
u1=fields[-2,:,::batchsize]         #距离下壁面0.0002
print('上壁面',u1)
b1=fields[19,:,::batchsize]           #距离下壁面9.5972e-5
print('中间位置',b1)

d11=pd.DataFrame(d1)
u11=pd.DataFrame(u1)
b11=pd.DataFrame(b1)


c_down=coords[0,:,::batchsize]
# print('坐标x',c_down)
c_down1=pd.DataFrame(c_down)


outfile='location_velocity_v.xlsx'
with pd.ExcelWriter(outfile) as writer:
    d11.to_excel(writer,sheet_name='下壁面',index=False)
    u11.to_excel(writer, sheet_name='上壁面', index=False)
    b11.to_excel(writer, sheet_name='中间', index=False)
print('保存了物理量excel')


outfile='location_coords_x.xlsx'
with pd.ExcelWriter(outfile) as writer:
    c_down1.to_excel(writer,sheet_name='coords_x',index=False)

print('保存了物理量excel')
