"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import numpy as np

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features."""
    def __init__(self):
        """Define the model layers."""
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# ─── Metrics Function ─────────────────────────────────────────────────────────
def compute_mae_r2(y_true, y_pred):
    y_true_np = y_true.detach().numpy()  #  Fixed
    y_pred_np = y_pred.detach().numpy()  # Fixed
    mae = np.mean(np.abs(y_true_np - y_pred_np))
    ss_res = np.sum((y_true_np - y_pred_np)**2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np))**2)
    r2 = 1 - ss_res/ss_tot
    return mae, r2

# ─── Main Script ──────────────────────────────────────────────────────────────
def main():
    # Load data
    df = pd.read_csv('data/housing.csv')
    print("Data shape:", df.shape)

    # Features and target
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    X_mean, X_std = X.mean(), X.std()
    X_scaled = (X - X_mean) / X_std

    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    loss_history = []

    for epoch in range(num_epochs):
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Save predictions
    with torch.no_grad():
        all_preds = model(X_tensor)
    df_out = pd.DataFrame({
        "actual": y_tensor.numpy().flatten(),
        "predicted": all_preds.numpy().flatten()
    })
    df_out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

if __name__ == "__main__":
    main()

