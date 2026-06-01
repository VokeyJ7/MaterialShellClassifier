import pandas as pd
import numpy as np
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Loading data and mapping labels
df = pd.read_csv("HollowShell_Dataset.csv", skiprows=1)
df.dropna(inplace=True)
df = df.iloc[:, 1:]
features = df.iloc[ : , :-1].to_numpy(dtype=np.float32)
labels = df.iloc[ : , -1]

label_mapping = {
    "Weak_Shell": 0,
   "Moderate_Shell": 1,
    "Optimal_Shell": 2
}
lst = ["Weak_Shell", "Moderate_Shell", "Optimal_Shell"]

y = df.iloc[:, -1].map(label_mapping).values


# Model architecture
class Model(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, conv_stride, pool_size, pool_stride, h1, h2):
    super().__init__()
    padding = (kernel_size - 1) // 2
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, conv_stride, padding)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, conv_stride, padding)
    self.relu2 = nn.ReLU()
    self.pool = nn.MaxPool1d(pool_size, pool_stride)

    self.avgPool = nn.AdaptiveAvgPool1d(1)
    self.flatten = nn.Flatten(start_dim=1)
    self.dense = nn.Sequential(
        nn.LazyLinear(h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 3)
    )

  def forward(self, x):
    fwd = self.conv1(x)
    fwd = self.relu1(fwd)

    fwd = self.conv2(fwd)
    fwd = self.relu2(fwd)
    fwd = self.pool(fwd)



    fwd = self.avgPool(fwd)
    fwd = self.flatten(fwd)

    out = self.dense(fwd)
    return out


# Dataset preprocessing
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

input_scaler = StandardScaler()

x_train = input_scaler.fit_transform(x_train)
x_test = input_scaler.transform(x_test)

xtr = torch.tensor(x_train, dtype=torch.float32)
xte = torch.tensor(x_test, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.long)
yte = torch.tensor(y_test, dtype=torch.long)

print(xtr.shape, ytr.shape)
print(xte.shape)

joblib.dump(input_scaler, "materialScaler")

xtr = xtr.unsqueeze(1)
xte = xte.unsqueeze(1)




class MaterialDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


train_dataset = MaterialDataset(xtr, ytr)
test_dataset = MaterialDataset(xte, yte)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Model training and validation
model = Model(in_channels=1, out_channels=64, kernel_size=7, conv_stride=3, pool_size=2, pool_stride=2, h1=20, h2=5)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)


def train_one_epoch():
  running_loss = 0.0
  model.train()
  for batch_idx, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")
    output = model(x_batch)
    loss = criterion(output, y_batch)
    running_loss += loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()

    if batch_idx % 3 == 0:
      avg_loss = running_loss / 100
      print(f"Training Batch:{batch_idx+1} Training Loss:{avg_loss: .2f}")
  print()

def test_one_epoch():
  model.train(False)
  running_loss = 0.0
  total_correct = 0
  total_samples = 0
  with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
      x_batch, y_batch = batch[0].to("cpu"), batch[1].to("cpu")
      output = model(x_batch)
      loss = criterion(output, y_batch)
      running_loss += loss.item()

      pred = torch.argmax(output, axis=1)
      total_correct += (pred == y_batch).sum().item()
      total_samples += y_batch.size(0)
    avg_loss = running_loss / len(test_loader)
    acc = total_correct / total_samples
    print(f"Validation Loss:{avg_loss: .2f}, Accuracy:{acc: .2f}")


epochs = 10
for i in range(epochs):
  train_one_epoch()
  test_one_epoch()



# Inference with user-input 
df.iloc[:, -1].value_counts(normalize=True)

torch.save(model.state_dict(), "MaterialModel.pth")

# Shell parameters 
# surface_area,	volume, curvature_mean,	curvature_gaussian,	thickness,	mass,	stress_max,	displacement_max,	compliance,	
# buckling_factor,	eigenvalue_1,	eigenvalue_2,	eigenvalue_3
input_params = [123.24, 1021.32, 0.682, 1.340, 28.345, 342.34, 2.9342, 8.234, 1.213, 3.324, 9.642, 0.783, 0.7832]
input_params = np.array(input_params)


input_params = input_params.reshape(1, -1)
output_scaler = StandardScaler()
input_scaled = output_scaler.fit_transform(input_params)

input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(1)

output = model(input_tensor)

pred = torch.argmax(output, axis=1)
print(pred)

shell_type = []
for key, value in label_mapping.items():
  int(label_mapping[key])
  if label_mapping[key] == pred:
    shell_type.append(key)

print(f"According to your parameters the predicted shell is a {shell_type[0].replace("_", " ")} type.")












