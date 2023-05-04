import numpy as np
import pandas as pd
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

start = time.time()

def mice_impute(data, max_iter=10, tol=1e-6):
    data_nan = data.copy()
    
    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    mice_imputer = IterativeImputer(max_iter=max_iter, tol=tol)
    df_imputed = mice_imputer.fit_transform(data_nan)

    # Only update NaN values
    for row, col in zip(*np.where(nan_mask)):
        data_nan.iat[row, col] = df_imputed[row, col]

    return data_nan

df_imputed = mice_impute(data)

# Convert the imputed data back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=data.columns)

df_imputed.to_csv("mice_result.csv", index=False)

end = time.time()

print("The time of execution of MICE is :", (end - start) * 10**3, "ms")
