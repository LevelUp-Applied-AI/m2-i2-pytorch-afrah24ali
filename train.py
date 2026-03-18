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
import matplotlib.pyplot as plt

# ─── Model Definition ─────────────────────────────────────────────────────────

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
    y_true_np = y_true.detach().numpy()  # ✅ Fixed
    y_pred_np = y_pred.detach().numpy()  # ✅ Fixed
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

    # Standardize features AND target ✅ Fixed the main issue!
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled.values, dtype=torch.float32)  # ✅ Scaled target
    print("X_tensor:", X_tensor.shape)
    print("y_tensor:", y_tensor.shape)

    # Train/test split
    N = len(X_tensor)
    idx = torch.randperm(N)
    train_idx = idx[:int(0.8*N)]
    test_idx = idx[int(0.8*N):]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_test, y_test = X_tensor[test_idx], y_tensor[test_idx]

    # Model, loss, optimizer
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    loss_history = []

    for epoch in range(num_epochs):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # Predictions for evaluation
    with torch.no_grad():
        train_preds = model(X_train)
        test_preds = model(X_test)

    # Metrics
    mae_train, r2_train = compute_mae_r2(y_train, train_preds)
    mae_test, r2_test = compute_mae_r2(y_test, test_preds)

    print(f"Train MAE: {mae_train:.2f}, R²: {r2_train:.4f}")
    print(f"Test MAE: {mae_test:.2f}, R²: {r2_test:.4f}")

    # Scatter plot: Actual vs Predicted (test set)
    plt.scatter(y_test.detach().numpy(), test_preds.detach().numpy())  # ✅ Fixed
    plt.plot([y_test.min().item(), y_test.max().item()], 
             [y_test.min().item(), y_test.max().item()], 'r--')  # ✅ Fixed
    plt.xlabel("Actual Price (scaled)")
    plt.ylabel("Predicted Price (scaled)")
    plt.title("Actual vs Predicted Prices (Test Set)")
    plt.savefig("predictions_plot.png")
    plt.close()

    # Loss curve
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()

    # Save predictions for full dataset ✅ Fixed syntax & detach
    with torch.no_grad():
        all_preds_scaled = model(X_tensor)
        # Convert predictions back to original scale
        all_preds = all_preds_scaled * y_std.values + y_mean.values

    df_out = pd.DataFrame({
        "actual": y.values.flatten(),
        "predicted": all_preds.numpy().flatten()
    })
    df_out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv, predictions_plot.png, and loss_curve.png")

if __name__ == "__main__":
    main()

