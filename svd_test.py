import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import random

def truncated_svd_impute(data, k, tol=1e-6, max_iter=200):
    data_nan = data.copy()
    
    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    imputer = SimpleImputer(strategy='mean')
    data_no_nan = imputer.fit_transform(data_nan)

    for _ in range(max_iter):
        svd = TruncatedSVD(n_components=k)
        reduced_data = svd.fit_transform(data_no_nan)
        data_reconstructed = svd.inverse_transform(reduced_data)

        diff = np.sqrt(np.nanmean((data_no_nan - data_reconstructed) ** 2))

        # Only update NaN values
        data_no_nan[nan_mask] = data_reconstructed[nan_mask]

        if diff < tol:
            break

    return data_no_nan

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

# Select 5 valid data points randomly
indices = []
while len(indices) < 5:
    row = random.randint(0, data.shape[0] - 1)
    col = random.randint(0, data.shape[1] - 1)
    if not np.isnan(data.iat[row, col]) and data.iat[row, col] != 0 and (row, col) not in indices:
        indices.append((row, col))

# Record original values and set them to NaN
original_values = [data.iat[row, col] for row, col in indices]
for row, col in indices:
    data.iat[row, col] = np.nan

# Perform Truncated SVD imputation
imputed_data = truncated_svd_impute(data, k=5)

# Retrieve imputed values and calculate the differences
imputed_values = [imputed_data[row, col] for row, col in indices]
diffs = np.abs(np.array(imputed_values) - np.array(original_values))

# Calculate accuracy only for non-zero original values
non_zero_mask = np.array(original_values) != 0
relative_diffs = diffs[non_zero_mask] / np.array(original_values)[non_zero_mask]
accuracy = 1 - np.mean(relative_diffs)

# Calculate the loss metrics
mae = np.mean(diffs)
mse = np.mean(diffs**2)

print(f"Accuracy: {accuracy:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
