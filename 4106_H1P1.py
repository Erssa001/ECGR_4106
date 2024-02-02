#%%
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
torch.set_printoptions(edgeitems=2, linewidth=75)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
#%%
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
    ]))
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
    ]))

#%%
class Net_1(nn.Module):
  def __init__(self, dropout_chance):
    super(Net_1, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32 * 32 * 3, 512)
    self.act1 = nn.Tanh()
    self.dropout1 = nn.Dropout(p=dropout_chance)
    self.fc2 = nn.Linear(512, 256)
    self.act2 = nn.Tanh()
    self.dropout2 = nn.Dropout(p=dropout_chance)
    self.fc3 = nn.Linear(256, 96)
    self.act3 = nn.Tanh()
    self.dropout3 = nn.Dropout(p=dropout_chance)
    self.fc4 = nn.Linear(96, 10)
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    out = self.flatten(x)
    out = self.act1(self.fc1(out))
    out = self.dropout1(out)
    out = self.act2(self.fc2(out))
    out = self.dropout2(out)
    out = self.act3(self.fc3(out))
    out = self.dropout3(out)
    out = self.log_softmax(self.fc4(out))
    return out
  
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def accuracy_calc(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())

    accuracy = correct / total  
    return accuracy

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        if ((epoch == 1) or (epoch % 1 == 0)):
            # Calculate training & validation accuracy
            train_accuracy = accuracy_calc(model, train_loader)
            val_accuracy = accuracy_calc(model, val_loader)
            print(f"Epoch {epoch}, Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
#%%
learning_rate = 0.01
n_epochs = 20
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
dropout_chance = 0.2

train_loader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=batch_size, shuffle=False)
model = Net_1(dropout_chance=dropout_chance).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

training_loop(
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    val_loader = val_loader
    )
#%%
train_loader = torch.utils.data.DataLoader(cifar10, batch_size= batch_size, shuffle=False)
print_model_parameters(model)

correct = 0
total = 0

with torch.no_grad():
  for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    _, predicted = torch.max(outputs, dim=1)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())

print("Model Training Accuracy: %f" % (correct / total))

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
  for imgs, labels in val_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    _, predicted = torch.max(outputs, dim=1)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())

    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
cnf_matrix = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Model Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Model Validation:")
print("Accuracy = %f" % (correct / total))
print("Precision =  %f" % (precision))
print("Recall =  %f" % (recall))
print("F1 Score = %f" % (2 * (precision * recall) / (precision + recall)))
# %%
