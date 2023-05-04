import numpy as np
import pandas as pd
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

start = time.time()

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

k = 5

imputed_data = truncated_svd_impute(data, k)

end = time.time()

# Convert imputed_data to a DataFrame before saving
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)
imputed_data_df.to_csv("svd_result.csv", index=False)

print("The time of execution of SVD is :", (end - start) * 10**3, "ms")
