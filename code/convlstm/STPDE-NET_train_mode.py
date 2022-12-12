from datasets import tqdm
from sklearn.metrics import mean_squared_error
from  input_data import *
from convlstm_model import *

model_weights = './convlstm_STPDE-NET_train_mode.pth'
torch.backends.cudnn.enabled = False

input_dim = 1

hidden_dim = (16,16,16)

kernel_size = (3, 3)

model = ConvLSTM(input_dim, hidden_dim,  kernel_size).cuda()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs =200
train_losses, valid_losses = [], []

best_score = float('inf')
labels1 = np.zeros((1184,12,15))
labels1 = np.expand_dims(labels1, axis=1)#preds.reshape(100,1,12,15)
Qnet_train1 = np.zeros((1184,1,7,12,15))
mld_train1 = np.zeros((1184,1,7,12,15))
u_train1 = np.zeros((1184,1,7,12,15))
v_train1 = np.zeros((1184,1,7,12,15))
sst1_train1 = np.zeros((1184,1,7,12,15))
T_d_train1 = np.zeros((1184,1,7,12,15))

u_d_train1 = np.zeros((1184,1,7,12,15))
labels2 = np.zeros((96,1,12,15))
Qnet_valid1 = np.zeros((96,1,7,12,15))
mld_valid1 = np.zeros((96,1,7,12,15))
u_valid1 = np.zeros((96,1,7,12,15))
v_valid1 = np.zeros((96,1,7,12,15))
sst1_valid1 = np.zeros((96,1,7,12,15))
T_d_valid1 = np.zeros((96,1,7,12,15))
u_d_valid1 = np.zeros((96,1,7,12,15))
pred_val= np.zeros((128,1,6,27))

print(labels1.shape)
sores = []
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

for epoch in range(epochs):
    print('Epoch: {}/{}'.format(epoch + 1, epochs))

    # train
    model.train()
    losses = 0
    loss1 = 0
    for i, data in tqdm(enumerate(trainloader)):
        data, label = data
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        data1 = data[:,0:3,:,:,:]
        out = model(data1)

        dT = torch.autograd.grad(outputs=out, inputs=data1, grad_outputs=torch.ones_like(out), retain_graph=True,create_graph=True)[0]

        dT_x = dT[:,1,:,:,:].reshape(32,1,7,6,27)
        dT_y = dT[:, 2, :, :, :].reshape(32,1,7,6,27)

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
        Qnet_train = data[:, 3, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        mld_train = data[:, 4, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_train = data[:, 5, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        v_train = data[:, 6, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_d_train = data[:, 8, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        sst1_train = data[:, 0, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        T_d_train = data[:, 7, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()

        Qnet_train2 = torch.Tensor(Qnet_train).cuda()
        mld_train2 = torch.Tensor(mld_train).cuda()
        u_train2 = torch.Tensor(u_train).cuda()
        v_train2 = torch.Tensor(v_train).cuda()
        u_d_train2 = torch.Tensor(u_d_train).cuda()
        sst1_train2 = torch.Tensor(sst1_train).cuda()
        T_d_train2 = torch.Tensor(T_d_train).cuda()
        label = label.reshape(32,1,6,27)

        aerfa = 0.1
        beta = 0.9
        pred1 = ((86400 * (Qnet_train2[:, :, 0, :, :] / (1025 * 4000 * mld_train2[:, :, 0, :, :])) - (u_train2[:, :, 0, :, :] * dT_x1) - (v_train2[:, :, 0, :, :] * dT_y1) - (u_d_train2[:, :, 0, :, :] * (sst1_train2[:, :, 0, :, :] - T_d_train2[:, :, 0, :, :]) / mld_train2[:, :, 0, :, :])) + sst1_train2[:, :, 0, :, :])
        pred2 = ((86400 * (Qnet_train2[:, :, 1, :, :] / (1025 * 4000 * mld_train2[:, :, 1, :, :])) - (u_train2[:, :, 1, :, :] * dT_x2) - (v_train2[:, :, 1, :, :] * dT_y2) - (u_d_train2[:, :, 1, :, :] * (sst1_train2[:, :, 1, :, :] - T_d_train2[:, :, 1, :, :]) / mld_train2[:, :, 1, :, :])) + sst1_train2[:, :, 1, :, :])
        pred3 = ((86400 * (Qnet_train2[:, :, 2, :, :] / (1025 * 4000 * mld_train2[:, :, 2, :, :])) - (u_train2[:, :, 2, :, :] * dT_x3) - (v_train2[:, :, 2, :, :] * dT_y3) - (u_d_train2[:, :, 2, :, :] * (sst1_train2[:, :, 2, :, :] - T_d_train2[:, :, 2, :, :]) / mld_train2[:, :, 2, :, :])) + sst1_train2[:, :, 2, :, :])
        pred4 = ((86400 * (Qnet_train2[:, :, 3, :, :] / (1025 * 4000 * mld_train2[:, :, 3, :, :])) - (u_train2[:, :, 3, :, :] * dT_x4) - (v_train2[:, :, 3, :, :] * dT_y4) - (u_d_train2[:, :, 3, :, :] * (sst1_train2[:, :, 3, :, :] - T_d_train2[:, :, 3, :, :]) / mld_train2[:, :, 3, :, :])) + sst1_train2[:, :, 3, :, :])
        pred5 = ((86400 * (Qnet_train2[:, :, 4, :, :] / (1025 * 4000 * mld_train2[:, :, 4, :, :])) - (u_train2[:, :, 4, :, :] * dT_x5) - (v_train2[:, :, 4, :, :] * dT_y5) - (u_d_train2[:, :, 4, :, :] * (sst1_train2[:, :, 4, :, :] - T_d_train2[:, :, 4, :, :]) / mld_train2[:, :, 4, :, :])) + sst1_train2[:, :, 4, :, :])
        pred6 = ((86400 * (Qnet_train2[:, :, 5, :, :] / (1025 * 4000 * mld_train2[:, :, 5, :, :])) - (u_train2[:, :, 5, :, :] * dT_x6) - (v_train2[:, :, 5, :, :] * dT_y6) - (u_d_train2[:, :, 5, :, :] * (sst1_train2[:, :, 5, :, :] - T_d_train2[:, :, 5, :, :]) / mld_train2[:, :, 5, :, :])) + sst1_train2[:, :, 5, :, :])
        pred7 = ((86400 * (Qnet_train2[:, :, 6, :, :] / (1025 * 4000 * mld_train2[:, :, 6, :, :])) - (u_train2[:, :, 6, :, :] * dT_x7) - (v_train2[:, :, 6, :, :] * dT_y7) - (u_d_train2[:, :, 6, :, :] * (sst1_train2[:, :, 6, :, :] - T_d_train2[:, :, 6, :, :]) / mld_train2[:, :, 6, :, :])) + sst1_train2[:, :, 6, :, :])
        pred8 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * pred1 + beta * pred2) + beta * pred3) + beta * pred4) + beta * pred5) + beta * pred6) + beta * pred7

        label = label.cpu()
        labels11 = torch.Tensor(label).cuda()
        loss = criterion(pred8, labels11)
        losses += loss

        loss.backward()
        optimizer.step()
    train_loss = losses / len(trainloader)
    train_losses.append(train_loss)
    print('Training Loss: {:.10f}'.format((train_loss)))

    # validation
    model.eval()
    losses = 0

    for i, data in tqdm(enumerate(validloader)):
        data, label = data
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        data1 = data[:,0:3,:,:,:]
        preds1 = model(data1)
        dT = torch.autograd.grad(outputs=preds1, inputs=data1, grad_outputs=torch.ones_like(preds1), retain_graph=True,create_graph=True)[0]  # dT_x:torch.Size([32, 3, 7, 12, 15])
        dT_x = dT[:, 1, :, :, :].reshape(32,1,7,6,27)
        dT_y = dT[:, 2, :, :, :].reshape(32,1,7,6,27)

        aerfa = 0.1
        beta = 0.9
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
        Qnet_valid = data[:, 3,:, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        mld_valid =  data[:, 4, :, :,:].reshape(-1, 1, 7, 6, 27).cpu()
        u_valid = data[:, 5, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        v_valid = data[:, 6, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_d_valid = data[:, 8, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        sst1_valid = data[:, 0, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        T_d_valid = data[:, 7, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()

        Qnet_valid2 = torch.Tensor(Qnet_valid).cuda()
        mld_valid2 = torch.Tensor(mld_valid).cuda()
        u_valid2 = torch.Tensor(u_valid).cuda()
        v_valid2 = torch.Tensor(v_valid).cuda()
        u_d_valid2 = torch.Tensor(u_d_valid).cuda()
        sst1_valid2 = torch.Tensor(sst1_valid).cuda()
        T_d_valid2 = torch.Tensor(T_d_valid).cuda()
        label = label.reshape(32,1,6,27)

        pred11 = ((84600 * (Qnet_valid2[:, :, 0, :, :] / (1025 * 4000 * mld_valid2[:, :, 0, :, :]))  - (u_valid2[:, :, 0, :, :] * dT_x1) - (v_valid2[:, :, 0, :, :] * dT_y1) - (u_d_valid2[:, :, 0, :, :] * (sst1_valid2[:, :, 0, :, :] - T_d_valid2[:, :, 0, :, :]) / mld_valid2[:, :, 0, :, :])) + sst1_valid2[:, :, 0, :, :])
        pred22 = ((84600 * (Qnet_valid2[:, :, 1, :, :] / (1025 * 4000 * mld_valid2[:, :, 1, :, :]))  - (u_valid2[:, :, 1, :, :] * dT_x2) - (v_valid2[:, :, 1, :, :] * dT_y2) - (u_d_valid2[:, :, 1, :, :] * (sst1_valid2[:, :, 1, :, :] - T_d_valid2[:, :, 1, :, :]) / mld_valid2[:, :, 1, :, :])) + sst1_valid2[:, :, 1, :, :])
        pred33 = ((84600 * (Qnet_valid2[:, :, 2, :, :] / (1025 * 4000 * mld_valid2[:, :, 2, :, :]))  - (u_valid2[:, :, 2, :, :] * dT_x3) - (v_valid2[:, :, 2, :, :] * dT_y3) - (u_d_valid2[:, :, 2, :, :] * (sst1_valid2[:, :, 2, :, :] - T_d_valid2[:, :, 2, :, :]) / mld_valid2[:, :, 2, :, :])) + sst1_valid2[:, :, 2, :, :])
        pred44 = ((84600 * (Qnet_valid2[:, :, 3, :, :] / (1025 * 4000 * mld_valid2[:, :, 3, :, :]))  - (u_valid2[:, :, 3, :, :] * dT_x4) - (v_valid2[:, :, 3, :, :] * dT_y4) - (u_d_valid2[:, :, 3, :, :] * (sst1_valid2[:, :, 3, :, :] - T_d_valid2[:, :, 3, :, :]) / mld_valid2[:, :, 3, :, :])) + sst1_valid2[:, :, 3, :, :])
        pred55 = ((84600 * (Qnet_valid2[:, :, 4, :, :] / (1025 * 4000 * mld_valid2[:, :, 4, :, :]))  - (u_valid2[:, :, 4, :, :] * dT_x5) - (v_valid2[:, :, 4, :, :] * dT_y5) - (u_d_valid2[:, :, 4, :, :] * (sst1_valid2[:, :, 4, :, :] - T_d_valid2[:, :, 4, :, :]) / mld_valid2[:, :, 4, :, :])) + sst1_valid2[:, :, 4, :, :])
        pred66 = ((84600 * (Qnet_valid2[:, :, 5, :, :] / (1025 * 4000 * mld_valid2[:, :, 5, :, :]))  - (u_valid2[:, :, 5, :, :] * dT_x6) - (v_valid2[:, :, 5, :, :] * dT_y6) - (u_d_valid2[:, :, 5, :, :] * (sst1_valid2[:, :, 5, :, :] - T_d_valid2[:, :, 5, :, :]) / mld_valid2[:, :, 5, :, :])) + sst1_valid2[:, :, 5, :, :])
        pred77 = ((84600 * (Qnet_valid2[:, :, 6, :, :] / (1025 * 4000 * mld_valid2[:, :, 6, :, :]))  - (u_valid2[:, :, 6, :, :] * dT_x7) - (v_valid2[:, :, 6, :, :] * dT_y7) - (u_d_valid2[:, :, 6, :, :] * (sst1_valid2[:, :, 6, :, :] - T_d_valid2[:, :, 6, :, :]) / mld_valid2[:, :, 6, :, :])) + sst1_valid2[:, :, 6, :, :])
        pred88 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * pred11 + beta * pred22) + beta * pred33) + beta * pred44) + beta * pred55) + beta * pred66) + beta * pred77

        label = label.cpu()
        labels22 = torch.Tensor(label).cuda()

        loss = criterion(pred88, labels22)
        losses += float(loss)
        pred88 = pred88.detach().cpu().numpy()
        pred_val[i * batch_size2:(i + 1) * batch_size2] = np.array(pred88)
    valid_loss = losses / len(validloader)
    valid_losses.append(valid_loss)

    valid_label1 = valid_label.reshape(-1,1).detach().cpu().numpy()
    preds1 = pred_val.reshape(-1,1)

    s = rmse(valid_label1,preds1)
    sores.append(s)
    print('Score: {:.3f}'.format(s))

    # save model
    if valid_loss < best_score:
        best_score = valid_loss
        checkpoint = {'best_score': valid_loss,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, model_weights)
        best_loss = valid_loss
        torch.save(model.state_dict(),
                   './convlstm_STPDE-NET_train_mode.pt')

print(sores)
print(best_score)
print(s)