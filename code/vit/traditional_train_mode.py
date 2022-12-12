from datasets import tqdm
from sklearn.metrics import mean_squared_error
from vit_model import *
from input_data_data_driven import *


model_weights1 = './vit_traditional_data-driven_train_mode.pth'



model = ViT(
                image_size = 256,
                patch_size = 32,
                num_classes = 1000,
                dim = 162,
                depth = 8,  #1
                heads = 8,  #2
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
).cuda()

model = model.cuda()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epochs = 200
train_losses, valid_losses = [], []

best_score = float('inf')
best_score1 = float('inf')
preds = np.zeros((128,6,27))
preds = np.expand_dims(preds, axis=1)

sores = []
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))

for epoch in range(epochs):
    print('Epoch: {}/{}'.format(epoch + 1, epochs))

    # train
    model.train()
    losses = 0
    for data, label in tqdm(trainloader):
        data = data.cuda()
        label = label.cuda()
        data = data.reshape(-1, 7, 7, 6, 27)  #（-1，49，6*27）
        n,b,c,h,w = data.size()
        data = data.reshape(n,b*c,h*w)
        optimizer.zero_grad()

        out = model(data)
        out = out.reshape(-1,1,6,27)
        loss = criterion(out, label)
        losses += loss

        loss.backward()
        optimizer.step()
    train_loss = losses / len(trainloader)
    train_losses.append(train_loss)
    print('Training Loss: {:.3f}'.format(train_loss))

    # validation
    # model.eval()
    losses = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader)):

            data, labels = data
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            data = data.reshape(-1, 7, 7, 6, 27)
            n, b, c, h, w = data.size()
            data = data.reshape(n, b * c, h * w)
            pred = model(data)
            pred = pred.reshape(-1,1,6,27)

            loss = criterion(pred, label)
            losses += loss

            preds[i * batch_size2:(i + 1) * batch_size2] = pred.cpu()

    valid_loss = losses / len(validloader)
    valid_losses.append(valid_loss)

    valid_label1 = valid_label.reshape(-1,1)
    preds1 = preds.reshape(-1,1)

    s = rmse(valid_label1,preds1)
    sores.append(s)
    print('Score: {:.3f}'.format(s))

    if valid_loss < best_score1:
        best_score1 = valid_loss
        checkpoint = {'best_score': valid_loss,
                              'state_dict': model.state_dict()}
        torch.save(checkpoint, model_weights1)
        best_loss = valid_loss
        torch.save(model.state_dict(),
                           './vit_traditional_data-driven_train_mode.pt')

print(sores)
print(best_score)
print(s)