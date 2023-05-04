import numpy as np
import pandas as pd
import time
from sklearn.impute import KNNImputer

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

start = time.time()

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

imputed_data = knn_impute(data)

# Convert the imputed data back to a DataFrame
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)

end = time.time()

imputed_data_df.to_csv("knnImputer_result.csv", index=False)

print("The time of execution of KNNImputer is :", (end - start) * 10**3, "ms")
