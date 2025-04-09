import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
scaler = joblib.load('scaler.pkl')  # ✅ Exact same scaler used for training

# 🔹 Load your deeper model definition
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        # First hidden layer: from input_size -> 1024
        self.fc1 = torch.nn.Linear(input_size, 1024)
        # Second hidden layer: from 1024 -> 512
        self.fc2 = torch.nn.Linear(1024, 512)
        # Third hidden layer: from 512 -> 256
        self.fc3 = torch.nn.Linear(512, 256)
        # Fourth hidden layer: from 256 -> 128
        self.fc4 = torch.nn.Linear(256, 128)
        # Output layer: from 128 -> 1
        self.fc5 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load the trained model
model = SimpleMLP(input_size=13)
model.load_state_dict(torch.load("mlp_restriction_predictor.pth", map_location=device))
model.to(device)
model.eval()

# 🔹 Refit your original MinMaxScaler using full training data
# Load X again if needed; this is the full unnormalized input array from training
# Example: X = np.load("X_train_data.npy")
# Here's a placeholder for X to show fitting the scaler:
# X = np.vstack([...])  # Should contain all training samples (4800, 5)

# Replace with your original training data if not already available
# Example dummy values (don't use in real case!)

# 🔹 Input sample(s) to test
# You can change this to any real test sample
sample_raw = np.array([
    [3, 241.6992188, -48.58852837, 24.30898592, 228.287547, 252.596533,	46.38671875, -28.16833292, 15.35618707, 33.31239914, 48.66858621, 35673.42023, 0.237822131],  # Sample 1
    [2, 231.9335938, -57.85246445, 36.29997657, 220.825057, 257.1250335, 21.97265625, -40.70497469, 10.6032475, 12.8682187, 23.4714662, 11351.18939, 0.075669674],
    [4, 375.9765625, -61.22049693, 31.218568, 355.4118717, 386.6304397, 21.97265625, -37.65657184, 1.796917977, 18.41740065, 20.21431863, 2285.443399, 0.015227426],  # Sample 3
    [5, 239.2578125, -53.94584967, 6.273890643, 235.6632955, 241.9371862, 46.38671875, -33.79851578, 14.87769571, 34.06569086, 48.94338657, 8605.966757, 0.057368449], #Sample 4 49
    [5, 239.2578125, -50.5015509, 11.4251032, 232.3047894, 243.7298926, 43.9453125, -31.82995239, 16.11277888, 33.78724902, 49.9000279, 13099.03207, 0.087301595], # Sample 5 56
    [2, 239.2578125, -55.36409288, -123.4323921, 226.6920779, 103.2596858, 21.97265625, -33.89358031, 4.425661932, 17.89541222, 22.32107415, 5629.292998, 0.037525671], # Sample 6 61
    [2, 302.734375, -52.46581523, 39.25745698, 275.6166302, 314.8740872, 21.97265625, -35.54354716, 3.159357661, 17.92419756, 21.08355522, 3005.881722, 0.020035128],# Sample 7 69
    [2, 241.6992188,-48.45838226, 11.24515439, 230.1502334, 241.3953878, 21.97265625, -30.99105786, 6.272740254, 17.95495914, 24.2276994, 7460.197269, 0.049731193], # # Sample 8 82
    [0, 229.0039063, -52.34740728, 46.71291928, 207.1983648, 253.9112841, 30.2734375, -31.18759181, 0.917848294, 29.08051981, 29.99836811, 2367.450879, 0.036257612]
])

# 🔹 Normalize
sample_scaled = scaler.transform(sample_raw)

# 🔹 Convert to tensor
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

# 🔹 Predict
with torch.no_grad():
    outputs = model(sample_tensor).cpu().numpy().flatten() * 100  # Convert to %

# 🔹 Display predictions
for i, pred in enumerate(outputs):
    print(f"Sample {i+1} ➤ Predicted Restriction: {pred:.2f}%")
