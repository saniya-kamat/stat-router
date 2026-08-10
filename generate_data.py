import pandas as pd
import numpy as np

np.random.seed(42) # generates same rndom num for each run
months = pd.date_range(start="2023-01-01", periods=24, freq="MS") # months from 2023-01-01 to 2024-12-01

# Directional: churn rate with a clear upward trend
churn = np.linspace(0.05, 0.15, 24) + np.random.normal(0, 0.01, 24) # combines trend + noise
df_directional = pd.DataFrame({"date": months, "churn_rate": churn})
df_directional.to_csv("data/churn_trend.csv", index=False) # automatically puts in data folder

# Comparative: avg order value, two segments with different means
northeast = np.random.normal(120, 15, 200) # normal distribution w/ mean 120, SD 15, and 200 values
southwest = np.random.normal(100, 18, 200)
df_comparative = pd.DataFrame({
    "segment": ["Northeast"] * 200 + ["Southwest"] * 200,
    "avg_order_value": np.concatenate([northeast, southwest])
})
df_comparative.to_csv("data/segment_comparison.csv", index=False)

# Causal: revenue with a clear level shift at month 12
pre = np.random.normal(50000, 3000, 12)
post = np.random.normal(62000, 3000, 12)
revenue = np.concatenate([pre, post])
df_causal = pd.DataFrame({"date": months, "revenue": revenue})
df_causal.to_csv("data/revenue_its.csv", index=False)

# Distributional: conversion rate, current week vs historical baseline
historical_mean = 0.08
historical_std = 0.01
current_value = 0.042  # anomalously low
df_distributional = pd.DataFrame({
    "metric": ["conversion_rate"],
    "current_value": [current_value],
    "historical_mean": [historical_mean],
    "historical_std": [historical_std]
})
df_distributional.to_csv("data/conversion_anomaly.csv", index=False)

print("Datasets generated successfully.")