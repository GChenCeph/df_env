import numpy as np
import pandas as pd
import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

start = time.time()

def bayesian_impute(data, tol=1e-6, max_iter=10):
    data_nan = data.copy()
    
    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    bayesian_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter, tol=tol)
    df_imputed = bayesian_imputer.fit_transform(data_nan)

    # Only update NaN values
    for row, col in zip(*np.where(nan_mask)):
        data_nan.iat[row, col] = df_imputed[row, col]

    return data_nan

df_imputed = bayesian_impute(data)

df_imputed = pd.DataFrame(df_imputed, columns=data.columns)

#np.savetxt("beyesian_result.csv", df_imputed, delimiter=",")
df_imputed.to_csv("beyesian_result.csv", index=False)

end = time.time()

print("The time of execution of Bayesian Regression is :", (end-start) * 10**3, "ms")
