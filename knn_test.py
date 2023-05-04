import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute(data, k=5):
    data_nan = data.copy()
    
    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    imputer = KNNImputer(n_neighbors=k)
    imputed_data = imputer.fit_transform(data_nan)

    # Only update NaN values
    for row, col in zip(*np.where(nan_mask)):
        data_nan.iat[row, col] = imputed_data[row, col]

    return data_nan

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

# Select 5 valid data points and set them to NaN
selected_points = [(0, 2), (5, 6), (15, 8), (35, 12), (50, 4)]
original_values = []

for row, col in selected_points:
    original_values.append(data.iat[row, col])
    data.iat[row, col] = np.NaN

# Perform KNN imputation
imputed_data = knn_impute(data)

# Find the imputed values for the selected points
imputed_values = [imputed_data.iat[row, col] for row, col in selected_points]

# Calculate the absolute differences and the accuracy
diffs = np.abs(np.array(imputed_values) - np.array(original_values))
relative_diffs = diffs / np.array(original_values)
accuracy = 1 - np.mean(relative_diffs)

print(f"Accuracy: {accuracy:.4f}")

# Calculate the loss metrics
mae = np.mean(diffs)
mse = np.mean(diffs**2)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
