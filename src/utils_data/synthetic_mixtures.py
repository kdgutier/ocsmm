import numpy as np
import pandas as pd


def gaussian_mixture(locs, p, size, scales=None):
    n_mixtures = len(p)
    d = len(locs[0])
    assert n_mixtures == len(p)
    if scales is None:
        scales = [np.identity(d)] * n_mixtures
    
    mixture_idxs = np.random.choice(n_mixtures,
                                    size=size,
                                    replace=True,
                                    p=p)
    
    mixture_idx, mixture_counts = np.unique(mixture_idxs, return_counts=True)
    
    #sample and mix
    k, mixture_sample = 0, np.zeros(shape=(size, d))
    for idx, mixture_size in zip(mixture_idx, mixture_counts):
        sample = np.random.multivariate_normal(mean=locs[idx],
                                               cov=scales,
                                               size=mixture_size)
        mixture_sample[k:k+mixture_size] = sample
        k+=mixture_size
    np.random.shuffle(mixture_sample)
    return mixture_sample

def generate_mixtures(n=100):
    # gaussian mixtures parameters
    locs = np.array([[-1, -1], [1, -1], [0, 1], [1, 1]])
    p1 = np.array([0.22, 0.64, 0.03, 0.11])
    p2 = np.array([0.22, 0.03, 0.64, 0.11])
    #p3 = np.array([0.61, 0.1, 0.06, 0.23])
    p3 = np.array([0.80, 0.01, 0.06, 0.13])
    p4 = np.array([0, 0, 0.06, 0.94])
    #scales = [0.15 * np.identity(2)] * 4
    scales = 0.07 * np.identity(2)
    n_samples = np.random.poisson(lam=100, size=n)
    
    df_dict = {'u_id': [], 'set': [], 'outlier': [], 'split':[]}

    assert n % 100 == 0, "provide multiples of 100"
    n1 = np.int(.50 * n)
    n2 = np.int(0.75 * n)
    n3 = np.int(0.97 * n)
    n4 = np.int(1.0 * n)

    # inlier source1
    for set_id in range(n1):
      x_set = gaussian_mixture(locs, p1, n_samples[set_id], scales)
      df_dict['u_id'].append(set_id)
      df_dict['set'].append(x_set)
      df_dict['outlier'].append(False)
      df_dict['split'].append('train')

    for set_id in range(n1, n2):
      x_set = gaussian_mixture(locs, p2, n_samples[set_id], scales)
      df_dict['u_id'].append(set_id)
      df_dict['set'].append(x_set)
      df_dict['outlier'].append(False)
      df_dict['split'].append('test')
    
    # inlier source2
    for set_id in range(n2, n3):
      x_set = gaussian_mixture(locs, p2, n_samples[set_id], scales)
      df_dict['u_id'].append(set_id)
      df_dict['set'].append(x_set)
      df_dict['outlier'].append(False)
      df_dict['split'].append('test')
    
    # outlier sources
    for set_id in range(n3, n4):
      #if (set_id % 2 == 0):
      #  x_set = gaussian_mixture(locs, p3, n_samples[set_id], scales)
      #elif (set_id % 2 == 1):
      #  x_set = gaussian_mixture(locs, p4, n_samples[set_id], scales)
      x_set = gaussian_mixture(locs, p3, n_samples[set_id], scales)
      df_dict['u_id'].append(set_id)
      df_dict['set'].append(x_set)
      df_dict['outlier'].append(True)
      df_dict['split'].append('test')
    
    df_dict['set'] = np.array(df_dict['set'])
    df_dict['outlier'] = np.array(df_dict['outlier'])
    df_dict['split'] = np.array(df_dict['split'])
    df = pd.DataFrame.from_dict(df_dict)
    return df

def generate_ts_mixture(n_dates, seed=None):
    if seed is not None:
      np.random.seed(seed=seed)
    ds = np.array(range(n_dates))

    # gaussian mixtures parameters
    locs = np.array([[-1, -1], [1, -1], [0, 1], [1, 1]])
    p1 = np.array([0.22, 0.64, 0.03, 0.11])
    p2 = np.array([0.22, 0.03, 0.64, 0.11])
    p3 = np.array([0.80, 0.01, 0.06, 0.13])
    p4 = np.array([0, 0, 0.06, 0.94])
    #scales = [0.15 * np.identity(2)] * 4
    scales = 0.15 * np.identity(2)
    n_samples = np.random.poisson(lam=100, size=n_dates)
    n_train = np.int(n_dates * 0.4)

    n_outliers = 5
    outlier_idx = np.random.choice(range(n_train, n_dates), n_outliers)
    outliers = np.isin(np.array(range(n_dates)), outlier_idx)

    sets, outliers, dss, split = [], [], [], []
    for ds in range(n_dates):
      # inlier sets
      if (ds % 2 == 0):
        x_set = gaussian_mixture(locs, p1, n_samples[ds], scales)
        outlier = False
      elif (ds % 2 == 1):
        x_set = gaussian_mixture(locs, p2, n_samples[ds], scales)
        outlier = False

      # outlier sets
      if (ds in outlier_idx):
        if (ds % 2 == 0):
          x_set = gaussian_mixture(locs, p3, n_samples[ds], scales)
          outlier = True
        else:
          x_set = gaussian_mixture(locs, p4, n_samples[ds], scales)
          outlier = True

      if ds <= n_train:
        split.append('train')
      else:
        split.append('validation')

      dss.append(ds)
      sets.append(x_set)
      outliers.append(outlier)

    ds = np.array(ds)
    sets = np.array(sets)
    outliers = np.array(outliers)
    split = np.array(split)
    return sets, outliers, dss, split

def generate_mixtures_panel(n_u_id, n_dates):
  df_dict = {'u_id': [], 'ds': [], 'split': [],'set': [], 'outlier': []}
  for u_id in range(n_u_id):
      sets, outliers, dss, split = generate_ts_mixture(n_dates, seed=u_id)
      df_dict['u_id'].append(u_id)
      df_dict['set'].append(sets)
      df_dict['outlier'].append(outliers)
      df_dict['ds'].append(dss)
      df_dict['split'].append(split)
    
  df = pd.DataFrame.from_dict(df_dict)
  return df
