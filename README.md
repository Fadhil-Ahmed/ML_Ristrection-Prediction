# 🫀 Heart Valve Restriction Prediction using Bandwidth Metrics (PyTorch)

This repository contains code and data processing for predicting heart valve restriction percentages from acoustic bandwidth metrics using a deep neural network built with PyTorch.

## 📁 Project Structure

- **Data**: 120 `.csv` files of the format `bandwidthMetrics_<restriction>_group<id>.csv`, each containing multiple rows of feature metrics.
- **Model**: A deep fully connected neural network with dropout layers trained to regress the restriction percentage.
- **Evaluation**: Trained model is evaluated on a test set and individual samples are predicted and displayed.

## 🔬 Input Features

Each row of the CSV files is used as a training sample with the following 13 features:

```
Distance, PeakFreq_f1, PeakVal_dB_f1, ThreeDbBW_f1, LeftFreq_f1,
RightFreq_f1, PeakFreq_f2, PeakVal_dB_f2, ThreeDbBW_f2, LeftFreq_f2,
RightFreq_f2, Energy, Variance
```

The label (target output) is the **restriction percentage**, extracted from the file name and normalized to a value between `0.0` and `1.0`.

## ⚙️ Model Architecture

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.dropout(torch.relu(self.fc4(x)))
        return self.fc5(x)
```

## 🧪 Training

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate = 0.001)
- **Epochs**: 100
- **Batch Size**: 64
- **Scaler**: `MinMaxScaler` used to normalize the 13 features.

The training loop outputs loss per epoch and saves the model and scaler after training.

## 📈 Inference

To predict restriction percentage on new samples:
1. Preprocess input with the saved `MinMaxScaler` (`scaler.pkl`)
2. Load model weights from `mlp_restriction_predictor.pth`
3. Predict and multiply output by 100 to return a percentage.

Example output:
```
Sample 1 ➤ Predicted Restriction: 48.67%
Sample 2 ➤ Predicted Restriction: 34.92%
...
```

## 💾 Files Saved
- `mlp_restriction_predictor.pth`: Trained model weights.
- `scaler.pkl`: Preprocessing scaler (MinMax) used during training.

## ✅ Usage Summary

To train:
```bash
python train_model.py
```

To test/predict on new samples:
```bash
python predict_samples.py
```

## 📌 Dependencies

- Python 3.x
- PyTorch
- scikit-learn
- pandas
- numpy
- joblib

---

**Researcher:** Fadhil Ahmed  
**Affiliation:** Trinity College – Department of Engineering  
**Project:** Acoustic Signal Analysis for Valve Restriction Estimation

