import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of all 120 CSV file paths
csv_files = [
    'bandwidthMetrics_0_group01.csv', 'bandwidthMetrics_0_group02.csv', 'bandwidthMetrics_0_group03.csv',
    'bandwidthMetrics_0_group04.csv', 'bandwidthMetrics_0_group05.csv', 'bandwidthMetrics_0_group06.csv',
    'bandwidthMetrics_0_group07.csv', 'bandwidthMetrics_0_group08.csv', 'bandwidthMetrics_0_group09.csv',
    'bandwidthMetrics_0_group10.csv', 'bandwidthMetrics_0_group11.csv', 'bandwidthMetrics_0_group12.csv',
    'bandwidthMetrics_0_group13.csv', 'bandwidthMetrics_0_group14.csv', 'bandwidthMetrics_0_group15.csv',
    'bandwidthMetrics_25_group01.csv', 'bandwidthMetrics_25_group02.csv', 'bandwidthMetrics_25_group03.csv',
    'bandwidthMetrics_25_group04.csv', 'bandwidthMetrics_25_group05.csv', 'bandwidthMetrics_25_group06.csv',
    'bandwidthMetrics_25_group07.csv', 'bandwidthMetrics_25_group08.csv', 'bandwidthMetrics_25_group09.csv',
    'bandwidthMetrics_25_group10.csv', 'bandwidthMetrics_25_group11.csv', 'bandwidthMetrics_25_group12.csv',
    'bandwidthMetrics_25_group13.csv', 'bandwidthMetrics_25_group14.csv', 'bandwidthMetrics_25_group15.csv',
    'bandwidthMetrics_34_group01.csv', 'bandwidthMetrics_34_group02.csv', 'bandwidthMetrics_34_group03.csv',
    'bandwidthMetrics_34_group04.csv', 'bandwidthMetrics_34_group05.csv', 'bandwidthMetrics_34_group06.csv',
    'bandwidthMetrics_34_group07.csv', 'bandwidthMetrics_34_group08.csv', 'bandwidthMetrics_34_group09.csv',
    'bandwidthMetrics_34_group10.csv', 'bandwidthMetrics_34_group11.csv', 'bandwidthMetrics_34_group12.csv',
    'bandwidthMetrics_34_group13.csv', 'bandwidthMetrics_34_group14.csv', 'bandwidthMetrics_34_group15.csv',
    'bandwidthMetrics_49_group01.csv', 'bandwidthMetrics_49_group02.csv', 'bandwidthMetrics_49_group03.csv',
    'bandwidthMetrics_49_group04.csv', 'bandwidthMetrics_49_group05.csv', 'bandwidthMetrics_49_group06.csv',
    'bandwidthMetrics_49_group07.csv', 'bandwidthMetrics_49_group08.csv', 'bandwidthMetrics_49_group09.csv',
    'bandwidthMetrics_49_group10.csv', 'bandwidthMetrics_49_group11.csv', 'bandwidthMetrics_49_group12.csv',
    'bandwidthMetrics_49_group13.csv', 'bandwidthMetrics_49_group14.csv', 'bandwidthMetrics_49_group15.csv',
    'bandwidthMetrics_56_group01.csv', 'bandwidthMetrics_56_group02.csv', 'bandwidthMetrics_56_group03.csv',
    'bandwidthMetrics_56_group04.csv', 'bandwidthMetrics_56_group05.csv', 'bandwidthMetrics_56_group06.csv',
    'bandwidthMetrics_56_group07.csv', 'bandwidthMetrics_56_group08.csv', 'bandwidthMetrics_56_group09.csv',
    'bandwidthMetrics_56_group10.csv', 'bandwidthMetrics_56_group11.csv', 'bandwidthMetrics_56_group12.csv',
    'bandwidthMetrics_56_group13.csv', 'bandwidthMetrics_56_group14.csv', 'bandwidthMetrics_56_group15.csv',
    'bandwidthMetrics_61_group01.csv', 'bandwidthMetrics_61_group02.csv', 'bandwidthMetrics_61_group03.csv',
    'bandwidthMetrics_61_group04.csv', 'bandwidthMetrics_61_group05.csv', 'bandwidthMetrics_61_group06.csv',
    'bandwidthMetrics_61_group07.csv', 'bandwidthMetrics_61_group08.csv', 'bandwidthMetrics_61_group09.csv',
    'bandwidthMetrics_61_group10.csv', 'bandwidthMetrics_61_group11.csv', 'bandwidthMetrics_61_group12.csv',
    'bandwidthMetrics_61_group13.csv', 'bandwidthMetrics_61_group14.csv', 'bandwidthMetrics_61_group15.csv',
    'bandwidthMetrics_69_group01.csv', 'bandwidthMetrics_69_group02.csv', 'bandwidthMetrics_69_group03.csv',
    'bandwidthMetrics_69_group04.csv', 'bandwidthMetrics_69_group05.csv', 'bandwidthMetrics_69_group06.csv',
    'bandwidthMetrics_69_group07.csv', 'bandwidthMetrics_69_group08.csv', 'bandwidthMetrics_69_group09.csv',
    'bandwidthMetrics_69_group10.csv', 'bandwidthMetrics_69_group11.csv', 'bandwidthMetrics_69_group12.csv',
    'bandwidthMetrics_69_group13.csv', 'bandwidthMetrics_69_group14.csv', 'bandwidthMetrics_69_group15.csv',
    'bandwidthMetrics_82_group01.csv', 'bandwidthMetrics_82_group02.csv', 'bandwidthMetrics_82_group03.csv',
    'bandwidthMetrics_82_group04.csv', 'bandwidthMetrics_82_group05.csv', 'bandwidthMetrics_82_group06.csv',
    'bandwidthMetrics_82_group07.csv', 'bandwidthMetrics_82_group08.csv', 'bandwidthMetrics_82_group09.csv',
    'bandwidthMetrics_82_group10.csv', 'bandwidthMetrics_82_group11.csv', 'bandwidthMetrics_82_group12.csv',
    'bandwidthMetrics_82_group13.csv', 'bandwidthMetrics_82_group14.csv', 'bandwidthMetrics_82_group15.csv'
]


# 🔁 Load all rows as individual samples
X_data = []
y_data = []

for file in csv_files:
    # Extract restriction percentage from filename (e.g., '34%' or '61.4%')
    restr_str = file.split('_')[1]
    restriction_value = float(file.split('_')[1]) / 100.0
    print(restriction_value)

    df = pd.read_csv(file)

    # Extract relevant features (13 columns)
    features = df[['Distance', 'PeakFreq_f1', 'PeakVal_dB_f1', 'ThreeDbBW_f1', 'LeftFreq_f1',
                   'RightFreq_f1', 'PeakFreq_f2', 'PeakVal_dB_f2', 'ThreeDbBW_f2', 'LeftFreq_f2',
                   'RightFreq_f2', 'Energy', 'Variance']].values

    # Create one label per row
    labels = np.full((features.shape[0], 1), restriction_value)

    X_data.append(features)
    y_data.append(labels)

# 🔄 Stack into single arrays
X = np.vstack(X_data)  # Shape: (4800, 5)
y = np.vstack(y_data)  # Shape: (4800, 1)

# Normalize input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 📦 PyTorch Dataset
class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# DataLoaders
batch_size = 64
train_loader = DataLoader(FlowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(FlowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)


# 🧠 Deeper Fully Connected Model
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        # First hidden layer: from input_size -> 1024
        self.fc1 = nn.Linear(input_size, 1024)
        # Second hidden layer: from 1024 -> 512
        self.fc2 = nn.Linear(1024, 512)
        # Third hidden layer: from 512 -> 256
        self.fc3 = nn.Linear(512, 256)
        # Fourth hidden layer: from 256 -> 128
        self.fc4 = nn.Linear(256, 128)
        # Output layer: from 128 -> 1
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.fc5(x)
        return x



# Init model
input_size = X_train.shape[1]  # Should be 5
model = SimpleMLP(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔁 Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

import joblib
joblib.dump(scaler, 'scaler.pkl')
# 💾 Save Model
torch.save(model.state_dict(), "mlp_restriction_predictor.pth")
print("\n✅ Model saved to: mlp_restriction_predictor.pth")

# 🔍 Testing
model.eval()
test_losses = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        test_losses.append(loss.item())
        print(
            f"🔸 Predicted: {(preds * 100).cpu().numpy().flatten()}, Actual: {(batch_y * 100).cpu().numpy().flatten()}")

print(f"\n✅ Final Test Loss (MSE): {np.mean(test_losses):.4f}")
