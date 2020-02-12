import numpy  as np
import pandas as pd


def generate_ts_data(n_dates, seed=None):
  if seed is not None:
    np.random.seed(seed=seed)
  ds = np.array(range(n_dates))
  
  #trend = dates.copy()
  trend = np.ones(n_dates)
  white_noise = np.random.normal(size=n_dates)

  n_outliers = 5
  outlier_idx = np.random.choice(range(n_dates), n_outliers)
  outlier_noise = np.random.standard_cauchy(size=n_outliers)*3 \
                  + np.random.choice([-2, 2], size=n_outliers, replace=True)

  y = trend + white_noise + 1
  y[outlier_idx] += outlier_noise

  return y, outlier_idx, ds

def generate_ts_panel(n_u_id, n_dates):
    
  df_dict = {'u_id': [], 'ds': [], 'y': []}
  for u_id in range(n_u_id):
    y, _, ds = generate_ts_data(n_dates, seed=u_id)
    df_dict['u_id'].append(u_id)
    df_dict['y'].append(y)
    df_dict['ds'].append(ds)
    
  df = pd.DataFrame.from_dict(df_dict)
  return df
