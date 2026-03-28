# experiment_tracker.py
import json
import matplotlib.pyplot as plt
from train import train_model

# Hyperparameter grid
hidden_sizes = [32, 64, 128]
learning_rates = [0.0005, 0.001, 0.005, 0.01]
epochs_list = [100, 200, 300]

configs = []
for h in hidden_sizes:
    for lr in learning_rates:
        for ep in epochs_list:
            configs.append({"hidden_size": h, "lr": lr, "epochs": ep})

results = []

for i, config in enumerate(configs, start=1):
    print(f"Running experiment {i}/{len(configs)}: {config}")
    res = train_model(config)
    results.append(res)

# Save all experiments
with open("experiments.json", "w") as f:
    results_serializable = []
for r in results:
    new_r = {
        "config": r["config"],
        "train_loss": float(r["train_loss"]),
        "test_loss": float(r["test_loss"]),
        "mae": float(r["mae"]),
        "r2": float(r["r2"]),
        "time": float(r["time"])
    }
    results_serializable.append(new_r)

with open("experiments.json", "w") as f:
    json.dump(results_serializable, f, indent=4)

# Leaderboard top 10 by Test MAE
results_sorted = sorted(results, key=lambda x: x["mae"])
print("\nTop 10 configurations by Test MAE:")
for i, r in enumerate(results_sorted[:10], start=1):
    c = r["config"]
    print(f"{i}. hidden={c['hidden_size']}, lr={c['lr']}, epochs={c['epochs']}, "
          f"Test MAE={r['mae']:.2f}, R²={r['r2']:.4f}, Time={r['time']:.1f}s")

# Summary visualization
plt.figure(figsize=(10,6))
for h in hidden_sizes:
    subset = [r for r in results if r["config"]["hidden_size"] == h]
    subset_sorted = sorted(subset, key=lambda x: x["config"]["lr"])
    lrs = [r["config"]["lr"] for r in subset_sorted]
    maes = [r["mae"] for r in subset_sorted]
    plt.plot(lrs, maes, marker='o', label=f"hidden={h}")

plt.xlabel("Learning Rate")
plt.ylabel("Test MAE")
plt.title("Hyperparameter Search: Test MAE vs Learning Rate")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.savefig("experiment_summary.png")
plt.show()