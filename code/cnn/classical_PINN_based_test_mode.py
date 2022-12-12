from datasets import tqdm
from sklearn.metrics import mean_squared_error
from cnn_model import *
from input_data import *

sores = []
criterion = nn.MSELoss()
preds1 = np.zeros((539,12,15))
preds1 = np.expand_dims(preds1, axis=1)
Qnet_test1 = np.zeros((539,1,7,12,15))
mld_test1 = np.zeros((539,1,7,12,15))
u_test1 = np.zeros((539,1,7,12,15))
v_test1 = np.zeros((539,1,7,12,15))
sst1_test1 = np.zeros((539,1,7,12,15))
T_d_test1 = np.zeros((539,1,7,12,15))
u_d_test1 = np.zeros((539,1,7,12,15))
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

checkpoint = torch.load(r'C:\Users\mymylove\Desktop\zege\开源代码_扩刊\code\cnn\cnn_classical_PINN-based_train_mode.pth')

model = ConvNet(input_dim=2,hidden_dim=1,kernel_size1=(3,3,3),padding1=(1,1,1),kernel_size2=(1,3,3),padding2=(0,1,1)).cuda()
model.load_state_dict(checkpoint['state_dict'])

# test
model.eval()
for k in range(1):
    for i, data in tqdm(enumerate(testloader)):
        data,label = data
        data = data.cuda()
        label = label.cuda()
        data1 = data[:,1:3,:,:,:]

        preds2 = model(data1)
        dT_x = torch.autograd.grad(outputs=preds2, inputs=xx_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]
        dT_y = torch.autograd.grad(outputs=preds2, inputs=yy_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]
        aerfa = 0.1
        beta = 0.9
        dT_x = dT_x[:1079]
        dT_y = dT_y[:1079]

        dT_x1 = dT_x[:, :, 0, :, :]
        dT_x2 = dT_x[:, :, 1, :, :]
        dT_x3 = dT_x[:, :, 2, :, :]
        dT_x4 = dT_x[:, :, 3, :, :]
        dT_x5 = dT_x[:, :, 4, :, :]
        dT_x6 = dT_x[:, :, 5, :, :]
        dT_x7 = dT_x[:, :, 6, :, :]
        dT_y1 = dT_y[:, :, 0, :, :]
        dT_y2 = dT_y[:, :, 1, :, :]
        dT_y3 = dT_y[:, :, 2, :, :]
        dT_y4 = dT_y[:, :, 3, :, :]
        dT_y5 = dT_y[:, :, 4, :, :]
        dT_y6 = dT_y[:, :, 5, :, :]
        dT_y7 = dT_y[:, :, 6, :, :]

        # Construction data
        Qnet_test = data[:, 3, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        mld_test = data[:, 4, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_test = data[:, 5, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        v_test = data[:, 6, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_d_test = data[:, 8, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        sst1_test = data[:, 0, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        T_d_test = data[:, 7, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()

        Qnet_test2 = torch.Tensor(Qnet_test)
        mld_test2 = torch.Tensor(mld_test)
        u_test2 = torch.Tensor(u_test)
        v_test2 = torch.Tensor(v_test)
        u_d_test2 = torch.Tensor(u_d_test)
        sst1_test2 = torch.Tensor(sst1_test)
        T_d_test2 = torch.Tensor(T_d_test)

    aerfa = 0.1
    beta = 0.9
    pred1 = ((86400 * (Qnet_test2[:, :, 0, :, :] / (1025 * 4000 * mld_test2[:, :, 0, :, :])) - (u_test2[:, :, 0, :, :] * dT_x1) - (v_test2[:, :, 0, :, :] * dT_y1) - (u_d_test2[:, :, 0, :, :] * (sst1_test2[:, :, 0, :, :] - T_d_test2[:, :, 0, :, :]) / mld_test2[:, :, 0, :, :])) + sst1_test2[:, :, 0, :, :]).cuda()
    pred2 = ((86400 * (Qnet_test2[:, :, 1, :, :] / (1025 * 4000 * mld_test2[:, :, 1, :, :])) - (u_test2[:, :, 1, :, :] * dT_x2) - (v_test2[:, :, 1, :, :] * dT_y2) - (u_d_test2[:, :, 1, :, :] * (sst1_test2[:, :, 1, :, :] - T_d_test2[:, :, 1, :, :]) / mld_test2[:, :, 1, :, :])) + sst1_test2[:, :, 1, :, :]).cuda()
    pred3 = ((86400 * (Qnet_test2[:, :, 2, :, :] / (1025 * 4000 * mld_test2[:, :, 2, :, :])) - (u_test2[:, :, 2, :, :] * dT_x3) - (v_test2[:, :, 2, :, :] * dT_y3) - (u_d_test2[:, :, 2, :, :] * (sst1_test2[:, :, 2, :, :] - T_d_test2[:, :, 2, :, :]) / mld_test2[:, :, 2, :, :])) + sst1_test2[:, :, 2, :, :]).cuda()
    pred4 = ((86400 * (Qnet_test2[:, :, 3, :, :] / (1025 * 4000 * mld_test2[:, :, 3, :, :])) - (u_test2[:, :, 3, :, :] * dT_x4) - (v_test2[:, :, 3, :, :] * dT_y4) - (u_d_test2[:, :, 3, :, :] * (sst1_test2[:, :, 3, :, :] - T_d_test2[:, :, 3, :, :]) / mld_test2[:, :, 3, :, :])) + sst1_test2[:, :, 3, :, :]).cuda()
    pred5 = ((86400 * (Qnet_test2[:, :, 4, :, :] / (1025 * 4000 * mld_test2[:, :, 4, :, :])) - (u_test2[:, :, 4, :, :] * dT_x5) - (v_test2[:, :, 4, :, :] * dT_y5) - (u_d_test2[:, :, 4, :, :] * (sst1_test2[:, :, 4, :, :] - T_d_test2[:, :, 4, :, :]) / mld_test2[:, :, 4, :, :])) + sst1_test2[:, :, 4, :, :]).cuda()
    pred6 = ((86400 * (Qnet_test2[:, :, 5, :, :] / (1025 * 4000 * mld_test2[:, :, 5, :, :])) - (u_test2[:, :, 5, :, :] * dT_x6) - (v_test2[:, :, 5, :, :] * dT_y6) - (u_d_test2[:, :, 5, :, :] * (sst1_test2[:, :, 5, :, :] - T_d_test2[:, :, 5, :, :]) / mld_test2[:, :, 5, :, :])) + sst1_test2[:, :, 5, :, :]).cuda()
    pred7 = ((86400 * (Qnet_test2[:, :, 6, :, :] / (1025 * 4000 * mld_test2[:, :, 6, :, :])) - (u_test2[:, :, 6, :, :] * dT_x7) - (v_test2[:, :, 6, :, :] * dT_y7) - (u_d_test2[:, :, 6, :, :] * (sst1_test2[:, :, 6, :, :] - T_d_test2[:, :, 6, :, :]) / mld_test2[:, :, 6, :, :])) + sst1_test2[:, :, 6, :, :]).cuda()
    pred8 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * pred1 + beta * pred2) + beta * pred3) + beta * pred4) + beta * pred5) + beta * pred6) + beta * pred7

pred_test1 = pred8.reshape(-1,1).detach().cpu().numpy()
test_label = test_label11.reshape(-1,1).detach().cpu().numpy()

s = rmse(test_label, pred_test1)
print('RMSE: {:.3f}'.format(s))

test_label = np.array(test_label)
pred_test1 = np.array(pred_test1)
MAE = np.mean(np.abs(test_label - pred_test1))
print('MAE: {:.3f}'.format(MAE))

MSE = np.mean(np.square(test_label - pred_test1))
print('MSE: {:.3f}'.format(MSE))

MAPE = np.mean(np.abs((test_label - pred_test1) / test_label))
print('MAPE: {:.3f}'.format(MAPE))

test_label1 = np.array((test_label))
pred_test1 = np.array(pred_test1)

x_hat = np.mean(test_label1)
x = test_label1
y_hat = np.mean(pred_test1)
y = pred_test1

fenzi = np.sum((x - x_hat) * (y - y_hat))
fenmu = np.sqrt((np.sum(np.square(x - x_hat))) * (np.sum(np.square(y - y_hat))))

cor = fenzi / fenmu
print(cor)

ture_1day = sst[valid_size + 0: 3639,:,:,:]
ture_2day = sst[valid_size + 1: 3640,:,:,:]
ture_3day = sst[valid_size + 2: 3641,:,:,:]
ture_4day = sst[valid_size + 3: 3642,:,:,:]
ture_5day = sst[valid_size + 4: 3643,:,:,:]
ture_6day = sst[valid_size + 5: 3644,:,:,:]
ture_7day = sst[valid_size + 6: 3645,:,:,:]
true_day = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * ture_1day + beta * ture_2day) + beta * ture_3day) + beta * ture_4day) + beta * ture_5day) + beta * ture_6day) + beta * ture_7day
ture_day = torch.tensor(true_day).cuda()

Tend_true = pred8 - ture_day




Qnet = aerfa*(aerfa*(aerfa*(aerfa*(aerfa*(aerfa*(86400*(Qnet_test2[:, :, 0, :, :] / (1025 * 4000 * mld_test2[:, :, 0, :, :]))) + beta*(86400*(Qnet_test2[:, :, 1, :, :] / (1025 * 4000 * mld_test2[:, :, 1, :, :])))) + beta*(86400*(Qnet_test2[:, :, 2, :, :] / (1025 * 4000 * mld_test2[:, :, 2, :, :])))
             ) + beta*(86400*(Qnet_test2[:, :, 3, :, :] / (1025 * 4000 * mld_test2[:, :, 3, :, :])))) + beta*((86400*(Qnet_test2[:, :, 4, :, :] / (1025 * 4000 * mld_test2[:, :, 4, :, :]))))) + beta*(86400*(Qnet_test2[:, :, 5, :, :] / (1025 * 4000 * mld_test2[:, :, 5, :, :])))) +beta*(86400*(Qnet_test2[:, :, 6, :, :] / (1025 * 4000 * mld_test2[:, :, 6, :, :])))

ZAdv = aerfa * (aerfa * (aerfa*(aerfa*(aerfa*(aerfa*((u_test2[:, :, 0, :, :] * dT_x1)) + beta*(u_test2[:, :, 1, :, :] * dT_x2)) + beta*(u_test2[:, :, 2, :, :] * dT_x3)) + beta*(u_test2[:, :, 3, :, :] * dT_x4))+ beta *(u_test2[:, :, 4, :, :] * dT_x5)
                ) + beta*(u_test2[:, :, 5, :, :] * dT_x6)) + beta*((u_test2[:, :, 6, :, :] * dT_x7))

MAdv = aerfa * (aerfa * (aerfa*(aerfa*(aerfa*(aerfa*((v_test2[:, :, 0, :, :] * dT_y1)) + beta*(v_test2[:, :, 1, :, :] * dT_y2)) + beta*(v_test2[:, :, 2, :, :] * dT_y3)) + beta*(v_test2[:, :, 3, :, :] * dT_y4))+ beta *(v_test2[:, :, 4, :, :] * dT_y5)
                ) + beta*(v_test2[:, :, 5, :, :] * dT_y6)) + beta*((v_test2[:, :, 6, :, :] * dT_y7))

VAdv = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * ((u_d_test2[:, :, 0, :, :] * (sst1_test2[:, :, 0, :, :] - T_d_test2[:, :, 0, :, :]) / mld_test2[:, :, 0, :, :])) + beta*(u_d_test2[:, :, 1, :, :] * (sst1_test2[:, :, 1, :, :] - T_d_test2[:, :, 1, :, :]) / mld_test2[:, :, 1, :, :])
                ) + beta * (u_d_test2[:, :, 2, :, :] * (sst1_test2[:, :, 2, :, :] - T_d_test2[:, :, 2, :, :]) / mld_test2[:, :, 2, :, :])) + beta * (u_d_test2[:, :, 3, :, :] * (sst1_test2[:, :, 3, :, :] - T_d_test2[:, :, 3, :, :]) / mld_test2[:, :, 3, :, :])
                ) + beta * (u_d_test2[:, :, 4, :, :] * (sst1_test2[:, :, 4, :, :] - T_d_test2[:, :, 4, :, :]) / mld_test2[:, :, 4, :, :])) + beta * (u_d_test2[:, :, 5, :, :] * (sst1_test2[:, :, 5, :, :] - T_d_test2[:, :, 5, :, :]) / mld_test2[:, :, 5, :, :])) + beta * (u_d_test2[:, :, 6, :, :] * (sst1_test2[:, :, 6, :, :] - T_d_test2[:, :, 6, :, :]) / mld_test2[:, :, 6, :, :])

Tend = Qnet - ZAdv - MAdv - VAdv

Qnet_1 = np.array(Qnet)
ZAdv_1 = np.array(ZAdv)
MAdv_1 = np.array(MAdv)
VAdv_1 = np.array(VAdv)
Tend1 = np.array(Tend)

Tend_true_1 = np.array(Tend_true)
print('Tend_no_R:{}'.format(np.mean(Tend1)))

print('Qnet:{}'.format(np.mean(Qnet_1)))

print('ZAdv:{}'.format(np.mean(ZAdv_1)))

print('MAdv:{}'.format(np.mean(MAdv_1)))

print('VAdv:{}'.format(np.mean(VAdv_1)))

print('Tend_true:{}'.format(np.mean(Tend_true_1)))