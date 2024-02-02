#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#%%
# Preprocessing here
df_train = pd.read_csv('./datasets/house-train.csv')
df_test = pd.read_csv('./datasets/house-test.csv')

# Check data type
pd.options.display.max_rows=90
df_dtype = pd.DataFrame(df_train.dtypes,columns=['dtype'])
print(df_dtype.value_counts())
df_dtype = pd.DataFrame(df_test.dtypes,columns=['dtype'])
print(df_dtype.value_counts())

# Features used to modeling
usefull_cols = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF'
                , 'FullBath', 'YearBuilt', 'YearRemodAdd', 'Fireplaces'
                ,'LotFrontage','WoodDeckSF','OpenPorchSF'
                ,'ExterQual','Neighborhood','MSZoning'
                ,'Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition',]
df_train_prepro = df_train[usefull_cols].copy()
df_test_prepro = df_test[usefull_cols].copy()

# Remove Nulls 
## GarageArea in test data
df_test_prepro['GarageArea'] = df_test_prepro['GarageArea'].fillna(df_train_prepro['GarageArea'].mean())
## TotalBsmtSF in test data
df_test_prepro['TotalBsmtSF'] = df_test_prepro['TotalBsmtSF'].fillna(df_train_prepro['TotalBsmtSF'].mean())

# One-hot encoding
df_train_prepro_1H = pd.get_dummies(df_train_prepro,columns=['Neighborhood','MSZoning','Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition'])
df_test_prepro_1H = pd.get_dummies(df_test_prepro,columns=['Neighborhood','MSZoning','Alley','LotShape','LandContour','Condition1','HouseStyle','MasVnrType','SaleCondition'])
#One-hot encoding: convert categorical data variables into a form that could be provided to machine learning.
#It creates binary (0 or 1) columns for each category in the original data.
#pd.get_dummies(): Tconverts categorical variable(s) into dummy/indicator variables.

df_train_prepro_1H = df_train_prepro_1H.replace({True: 1, False: 0})
df_test_prepro_1H = df_test_prepro_1H.replace({True: 1, False: 0})
df_train_prepro = df_train_prepro.replace({True: 1, False: 0})
df_test_prepro = df_test_prepro.replace({True: 1, False: 0})

# Convert all columns to numeric (float) and handle NaN values
df_train_prepro = df_train_prepro.apply(pd.to_numeric, errors='coerce').fillna(0)
df_test_prepro = df_test_prepro.apply(pd.to_numeric, errors='coerce').fillna(0)
df_train_prepro_1H = df_train_prepro_1H.apply(pd.to_numeric, errors='coerce').fillna(0)
df_test_prepro_1H = df_test_prepro_1H.apply(pd.to_numeric, errors='coerce').fillna(0)

# Save the DataFrame to a CSV file
output_file = ['datasets/temp/housing_df_train_prepro.csv', 'datasets/temp/housing_df_test_prepro.csv', 'datasets/temp/housing_df_train_prepro_1H.csv', 'datasets/temp/housing_df_test_prepro_1H.csv']
df_train_prepro_1H.to_csv(output_file[2], index=False)
print(f'DataFrame saved to {output_file[2]}')
df_dtype = pd.DataFrame(df_train_prepro_1H.dtypes,columns=['dtype'])
print(df_dtype.value_counts())

df_test_prepro_1H.to_csv(output_file[3], index=False)
print(f'DataFrame saved to {output_file[3]}')
df_dtype = pd.DataFrame(df_test_prepro_1H.dtypes,columns=['dtype'])
print(df_dtype.value_counts())

df_train_prepro.to_csv(output_file[0], index=False)
print(f'DataFrame saved to {output_file[0]}')
df_dtype = pd.DataFrame(df_train_prepro.dtypes,columns=['dtype'])
print(df_dtype.value_counts())

df_test_prepro.to_csv(output_file[1], index=False)
print(f'DataFrame saved to {output_file[1]}')
df_dtype = pd.DataFrame(df_test_prepro.dtypes,columns=['dtype'])
print(df_dtype.value_counts())

# Scaling
scaler = StandardScaler()
X_data = torch.tensor(scaler.fit_transform(df_train_prepro.values_1H.copy()), dtype=torch.float32)
Y_data = torch.tensor(df_train['SalePrice'].values.copy(), dtype=torch.float32)

# Training/Test Split
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

#%%
class Net_1(nn.Module):
  def __init__(self, x_shape, dropout_chance):
    super(Net_1, self).__init__()
    self.fc1 = nn.Linear(x_shape, 1024)
    self.dropout1 = nn.Dropout(p=dropout_chance)
    self.act1 = nn.ReLU()
    self.fc2 = nn.Linear(1024, 512)
    self.dropout2 = nn.Dropout(p=dropout_chance)
    self.act2 = nn.ReLU()
    self.fc3 = nn.Linear(512, 256)
    self.dropout3 = nn.Dropout(p=dropout_chance)
    self.act3 = nn.ReLU()
    self.fc4 = nn.Linear(256, 1)

  def forward(self, x):
    out = self.act1(self.fc1(x))
    out = self.dropout1(out)
    out = self.act2(self.fc2(out))
    out = self.dropout2(out)
    out = self.act3(self.fc3(out))
    out = self.dropout3(out)
    out = self.fc4(out)
    return out.squeeze()
  
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, test_loader):
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += np.sqrt(loss.item())


        if ((epoch == 1) or (epoch % 1 == 0)):
            print(f"Epoch {epoch}, Training Loss: {train_loss/len(train_loader):.4f}, Validation Loss: {validation_loop(model=model, loss_fn=loss_fn, test_loader=test_loader)}")

def validation_loop(model, loss_fn, test_loader):
    model.eval()
    val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, target)
            val_loss += np.sqrt(loss.item())
            total_samples += features.size(0)

    avg_val_loss = val_loss / len(test_loader)
    return f"Validation Loss: {avg_val_loss:.4f}"
            
#%%
learning_rate = .01
n_epochs = 20
batch_size = 1
dropout_chance = 0.25
loss_fn = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
model = Net_1(x_shape=x_train.shape[1], dropout_chance=dropout_chance).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
training_loop(
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    test_loader = test_loader
    )
print_model_parameters(model)
