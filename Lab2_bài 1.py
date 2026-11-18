#Bài 1: Làm sạch dữ liệu
import numpy as np
import pandas as pd

# 1. Create a DataFrame with missing values (NaN) and outliers
# Using enough normal data points ensures the outlier exceeds 3 standard deviations
data = {
    'Age': [
        25,
        28,
        30,
        np.nan,
        27,
        29,
        31,
        26,
        28,
        30,
        29,
        27,
        150,
    ],  # 150 is an outlier
    'Salary': [
        50000,
        52000,
        48000,
        51000,
        np.nan,
        53000,
        49000,
        50500,
        51500,
        52500,
        49500,
        51000,
        500000,
    ],  # 500000 is an outlier
}

df = pd.DataFrame(data)
print('--- Original DataFrame ---')
print(df)

# 2. Fill missing values (NaN) with the mean of the respective column
df_filled = df.fillna(df.mean())
print('\n--- After Filling Missing Values with Column Mean ---')
print(df_filled.round(2))

# 3. Detect and handle outliers (> 3 standard deviations from the mean)
# Calculate Z-score for each column: |x - mean| / std
z_scores = np.abs((df_filled - df_filled.mean()) / df_filled.std())

# Identify rows where all columns are within 3 standard deviations
df_cleaned = df_filled[(z_scores <= 3).all(axis=1)]

print('\n--- Final Cleaned DataFrame (Outliers Removed) ---')
print(df_cleaned.round(2))
