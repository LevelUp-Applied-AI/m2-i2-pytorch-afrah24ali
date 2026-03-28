[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

This assignment implements a neural network to predict **apartment prices in Jordanian Dinars (JOD)** from a small tabular dataset.

---

## Model Objective

The model predicts `price_jod` using 5 input features:

- `area_sqm` — Apartment area in square meters (50–250)  
- `bedrooms` — Number of bedrooms (1–5)  
- `floor` — Floor number (1–15)  
- `age_years` — Building age in years (0–40)  
- `distance_to_center_km` — Distance to city center in km (0.5–25)  

---

## Training Configuration

- **Model architecture:** 5 → 32 (ReLU) → 1  
- **Loss function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Learning rate:** 0.01  
- **Number of epochs:** 100  

---

## Training Outcome

- Loss decreased gradually over epochs.  
- Final loss value: approximately **1.94e9** (due to the large scale of prices).  

---

## Behavioral Observation

- Initial loss was very high because the target variable `price_jod` was not standardized.  
- Loss decreased more quickly in early epochs and then slowed.  
- Predictions tend to slightly underestimate very high-priced apartments (above ~120,000 JOD).  
