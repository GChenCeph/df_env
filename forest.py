import numpy as np
import pandas as pd
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

start = time.time()

def random_forest_impute(data, n_estimators=10, random_state=0, max_iter=10, tol=1e-6):
    data_nan = data.copy()
    
    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    random_forest_imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        max_iter=max_iter,
        tol=tol,
    )
    df_imputed = random_forest_imputer.fit_transform(data_nan)

    # Only update NaN values
    for row, col in zip(*np.where(nan_mask)):
        data_nan.iat[row, col] = df_imputed[row, col]

    return data_nan

df_imputed = random_forest_impute(data)

# Convert the imputed data back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=data.columns)

df_imputed.to_csv("forest_result.csv", index=False)

end = time.time()

print("The time of execution of Random Forest is :", (end - start) * 10**3, "ms")
