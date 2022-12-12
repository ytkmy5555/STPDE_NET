import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setting random seed
setup_seed(1) # range(1,6)

append_day = 7  # range(8,17)

data1 = np.load('./label.npz')

sst = data1['sst'][:].reshape(-1, 1, 6, 27)
mld = data1['mld'][:].reshape(-1, 1, 6, 27)

data2 = np.load('./solar_radiation.npz')
print(data2.files)
dlwrf1 = data2['dlwrf1'][:]
dswrf1 = data2['dswrf1'][:]
lhtfl1 = data2['lhtfl1'][:]
shtfl1 = data2['shtfl1'][:]
ulwrf1 = data2['ulwrf1'][:]
uswrf1 = data2['uswrf1'][:]

data3 = np.load('./deep_variables.npz')
T_d1 = data3['T_d1'][:]
v_d1 = data3['v_d1'][:]
u_d1 = data3['u_d1'][:]

data4 = np.load('./other_variables.npz')
mld1 = data4['mld1'][:]
sst_feature = data4['sst_feature'][:]
vflx1 = data4['vflx1'][:]
uflx1 = data4['uflx1'][:]

data5 = np.load('./coordinate_variables.npz')
xx1 = data5['xx1'][:]
yy1 = data5['yy1'][:]

Qshortwave = dswrf1 - uswrf1
Qlongwave = dlwrf1 - ulwrf1

train_size = 2432
valid_size = 2560

u = uflx1.reshape(-1, 1, 7, 6, 27)
u = torch.Tensor(u)
u_train = u[0:train_size,:,:,:,:]
u_valid = u[train_size:valid_size,:,:,:,:]
u_test = u[valid_size:3646,:,:,:,:]

v = vflx1.reshape(-1, 1, 7, 6, 27)
v = torch.Tensor(v)
v_train = v[0:train_size,:,:,:,:]
v_valid = v[train_size:valid_size,:,:,:,:]
v_test = v[valid_size:3646,:,:,:,:]

x = np.arange(4.76184, -4.76184, -1.9047)
y = np.arange(191.25, 241.1, 1.875).reshape(-1, 1)
yy, xx = np.meshgrid(y, x)
xx = np.expand_dims(xx, 0).repeat(3652, axis=0)
yy = np.expand_dims(yy, 0).repeat(3652, axis=0)

datax15 = []
for i in range(3646):
    datax15.append(xx[i:i+7])
    xx1 = np.array(datax15)

datax16 = []
for i in range(3646):
    datax16.append(yy[i:i+7])
    yy1 = np.array(datax16)

xx = xx1.reshape(-1, 1, 7, 6, 27)
xx = torch.Tensor(xx)
xx_train = xx[0:train_size,:,:,:,:]
xx_valid = xx[train_size:valid_size,:,:,:,:]
xx_test = xx[valid_size:3646,:,:,:,:]

yy = yy1.reshape(-1, 1, 7, 6, 27)
yy = torch.Tensor(yy)
yy_train = yy[0:train_size,:,:,:,:]
yy_valid = yy[train_size:valid_size,:,:,:]
yy_test = yy[valid_size:3646,:,:,:]
xx_train.requires_grad = True
xx_valid.requires_grad = True
xx_test.requires_grad = True
xx.requires_grad = True
yy.requires_grad = True
yy_train.requires_grad = True
yy_valid.requires_grad = True
yy_test.requires_grad = True

Qnet = Qshortwave + Qlongwave + lhtfl1 + shtfl1
Qnet = Qnet.reshape(-1, 1, 7, 6, 27)
Qnet = torch.Tensor(Qnet)
Qnet_train = Qnet[0:train_size,:,:,:,:]
Qnet_valid = Qnet[train_size:valid_size,:,:,:,:]
Qnet_test = Qnet[valid_size:3646,:,:,:,:]

mld = mld1.reshape(-1, 1, 7, 6, 27)
mld = torch.Tensor(mld)
mld_train = mld[0:train_size,:,:,:]
mld_valid = mld[train_size:valid_size,:,:,:]
mld_test = mld[valid_size:3646,:,:,:]

sst1 = sst_feature.reshape(-1, 1, 7, 6, 27)
sst1 = torch.Tensor(sst1)
sst1_train = sst1[0:train_size,:,:,:,:]
sst1_valid = sst1[train_size:valid_size,:,:,:,:]
sst1_test = sst1[valid_size:3646,:,:,:,:]

T_d = T_d1.reshape(-1, 1, 7, 6, 27)
T_d = torch.Tensor(T_d)
T_d_train = T_d[0:train_size,:,:,:,:]
T_d_valid = T_d[train_size:valid_size,:,:,:,:]
T_d_test = T_d[valid_size:3646,:,:,:,:]

u_d = u_d1.reshape(-1, 1, 7, 6, 27)
u_d = torch.Tensor(u_d)
u_d_train = u_d[0:train_size,:,:,:,:]
u_d_valid = u_d[train_size:valid_size,:,:,:,:]
u_d_test = u_d[valid_size:3646,:,:,:,:]

v_d = v_d1.reshape(-1, 1, 7, 6, 27)
v_d = torch.Tensor(v_d)
v_d_train = v_d[0:train_size,:,:,:,:]
v_d_valid = v_d[train_size:valid_size,:,:,:,:]
v_d_test = v_d[valid_size:3646,:,:,:,:]

train_data = torch.cat((sst1_train, xx_train, yy_train,Qnet_train,mld_train,u_train,v_train,T_d_train,v_d_train), dim=1)

valid_data = torch.cat((sst1_valid, xx_valid, yy_valid,Qnet_valid,mld_valid,u_valid,v_valid,T_d_valid,v_d_valid), dim=1)

test_data = torch.cat((sst1_test, xx_test, yy_test,Qnet_test,mld_test,u_test,v_test,T_d_test,v_d_test), dim=1)

train_label = sst[append_day:train_size + append_day,:,:,:]

valid_label = sst[train_size + append_day:valid_size + append_day,:,:,:]


test_label = sst[valid_size + append_day: 3646,:,:,:]


train_label = torch.Tensor(train_label)
valid_label = torch.Tensor(valid_label)
test_label = torch.Tensor(test_label)

test_label11 = test_label
valid_label11 = valid_label

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

batch_size1 = 32
batch_size2 = 32
batch_size3 = 3000

trainset = MyDataset(train_data, train_label)
trainloader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)

validset = MyDataset(valid_data, valid_label)
validloader = DataLoader(validset, batch_size=batch_size2, shuffle=True, drop_last=False,pin_memory=True, num_workers=0)

testset = MyDataset(test_data, test_label)
testloader = DataLoader(testset, batch_size=batch_size3, shuffle=False, drop_last=False,pin_memory=True, num_workers=0)