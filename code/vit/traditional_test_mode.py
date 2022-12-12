from datasets import tqdm
from sklearn.metrics import mean_squared_error
from vit_model import *
from input_data import *

sores = []
preds1 = np.zeros((539,12,15))
preds1 = np.expand_dims(preds1, axis=1)

def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

checkpoint = torch.load('./vit_traditional_data-driven_train_mode.pth')

model = ViT(image_size = 256,patch_size = 32,num_classes = 1000,dim = 162,depth = 8,heads = 8,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1)
model.load_state_dict(checkpoint['state_dict'])
plot_pred = []
plot_label = []

# test

model.eval()
for i, data in tqdm(enumerate(testloader)):
    data, label = data
    data = data
    n, b, c, h, w = data.size()
    data = data.reshape(n, b * c, h * w)

    pred_test = model(data)

    label = label

pred_test1 = pred_test.reshape(-1,1).detach().numpy()
test_label = label.reshape(-1,1).detach().numpy()


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