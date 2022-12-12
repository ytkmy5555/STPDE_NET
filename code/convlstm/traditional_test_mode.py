from datasets import tqdm
from sklearn.metrics import mean_squared_error
from convlstm_model import *
from input_data import *

sores = []
criterion = nn.MSELoss()
preds1 = np.zeros((539,12,15))
preds1 = np.expand_dims(preds1, axis=1)

def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

checkpoint = torch.load('./convlstm_traditional_data-driven_train_mode.pth')

input_dim = 1

hidden_dim = (32,32,32)

kernel_size = (3, 3)

model =  ConvLSTM(input_dim, hidden_dim,  kernel_size)

model.load_state_dict(checkpoint['state_dict'])

# test

losses = 0
model.eval()
plot_pred = []
plot_label = []
for i, data in tqdm(enumerate(testloader)):
    print(i)
    data, label = data
    data = data
    label = label
    data = data.reshape(-1, 7, 7, 6, 27)
    pred_test = model(data)
    pred_test = pred_test.reshape(-1,1,6,27)

    test_label = np.array(test_label)

pred_test1 = pred_test.reshape(-1,1)
test_label = test_label.reshape(-1,1)

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