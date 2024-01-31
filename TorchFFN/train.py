from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from FFN import Classification_NN

classification_dataset = datasets.load_breast_cancer()
X, y = classification_dataset.data, classification_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(f'Training examples : {X_train.shape[0]}')
print(f'Test examples : {X_test.shape[0]}')

n_samples, n_features = X_train.shape
model = Classification_NN(input_dim=n_features, hidden_dim=3)
print(sum(p.numel() for p in model.parameters()),' parameters to train')
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005)

n_epochs = 500
for i in range(n_epochs):
    loss, out = model(torch.tensor(X_train).float(),torch.tensor(y_train).float().view(-1,1))
    # Print loss every once in a while
    if i % 50 == 0 or i == n_epochs-1:
        print(f'Epoch : {i} => Loss : {loss.item() : .2f}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()
loss, out = model(torch.tensor(X_test).float(),torch.tensor(y_test).float().view(-1,1))
print(f'Test loss = {loss.item() : .2f}')
pred_y = torch.where(out > 0.5, 1.0, 0.0)
acc = (torch.sum(pred_y == torch.tensor(y_test).float().view(-1,1))/pred_y.shape[0]).item()
print(f'Test Accuracy = {acc : .2f}')