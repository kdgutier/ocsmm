import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import spines
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.metrics import roc_curve

# import platform
# if platform.system() == 'Darwin':
#     matplotlib.use('MacOSX')
# else:
#     matplotlib.use('TkAgg')


def plot_residuals_outliers(y, y_hat, outlier, outlier_score, threshold):
  n_dates = len(y)
  ds = ds = np.array(range(n_dates))
  
  fig, axs = plt.subplots(1, 2, figsize=(4*2, 3.5))
  
  axs[0].plot(ds, y, color='C0', label='y')
  axs[0].plot(ds, y_hat, color='orange', linestyle='--', label='y_hat')
  
  if len(outlier)>0:
    axs[0].scatter(ds[outlier], y[outlier], 
                   color='red', label='outlier')
  
  axs[0].set_xlabel('ds')
  axs[0].set_ylabel('value')
  axs[0].legend()
  
  axs[1].plot(ds, outlier_score, c='C0')
  axs[1].axhline(threshold[0], c='red')
  axs[1].axhline(threshold[1], c='red')
  axs[1].set_xlabel('ds')
  axs[1].set_ylabel('Normalized Residuals')
  
  plt.savefig('./results/plots/residual_outlier.png')
  plt.tight_layout()
  plt.show()

def plot_mixtures(distributions_df):
  plot_df = distributions_df.copy()
  plot_df = plot_df.reset_index()
  plt.figure(figsize=(20,10))
  for idx in range(50):
    ax = plt.subplot(5, 10, idx + 1)
    if idx > 46:
      ax.spines['bottom'].set_color('green')
      ax.spines['bottom'].set_linewidth(5)
      ax.spines['top'].set_color('green')
      ax.spines['top'].set_linewidth(5)
      ax.spines['right'].set_color('green')
      ax.spines['right'].set_linewidth(5)
      ax.spines['left'].set_color('green')
      ax.spines['left'].set_linewidth(5)
    
    outlier = plot_df.loc[idx, 'outlier']
    distrib = plot_df.loc[idx, 'set']
    
    c = 'C0'
    if outlier == True: c = 'r'
    ax.scatter(distrib[:, 0], distrib[:, 1], c=c)
      
  plt.tight_layout()
  plt.show()
  plt.savefig('./results/plots/mixture_scatter.png')

def plot_roc_curve(y_test, y_hat):
  plt.figure(figsize=(4, 3.5))
  fpr, tpr, th = roc_curve(y_test, y_hat)
  plt.plot(fpr, tpr)
  plt.plot([0.2 * i for i in range(0, 6)],
            [0.2 * i for i in range(0, 6)])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.tight_layout()
  plt.show()
  plt.savefig('./results/plots/roc_curve.png')

def plot_roc_curves(y_test, y_hat_dict, log_scale=False):
  fig, ax = plt.subplots(1, figsize=(4, 3.5))
  plt.subplots_adjust(wspace=0.35)

  colors = ['C0', 'orange', 'green']
  linestyles = ['-', '--', ':']
  for idx, model_name in enumerate(y_hat_dict.keys()):
    y_hat = y_hat_dict[model_name]
    fpr, tpr, th = roc_curve(y_test, y_hat)
    plt.plot(fpr, tpr, linestyle=linestyles[idx],
             label=model_name, c=colors[idx])

  # Chance
  x_chance = np.linspace(0.00005, 1, 1000, endpoint=True)
  y_chance = np.linspace(0.00005, 1, 1000, endpoint=True)
  plt.plot(x_chance, y_chance, linestyle='-',
           label='Chance', color='black')

  if log_scale:
    plt.xscale('log')
    plt.xlabel('False Positive Rate (log)')
  else:
    plt.xlabel('False Positive Rate')

  plt.ylabel('True Positive Rate')
  plt.legend()
  plt.tight_layout()
  plt.show()
  plt.savefig("./results/plots/roc_curves.png", bbox_inches = "tight", dpi=300)
  

def plot_sample_size_times(ns, ocsmm_times, nocsmm_times, onocsmm_times):
  xs = np.array(ns)
  
  plt.figure(figsize=(4, 3.5))
  
  plt.plot(xs, ocsmm_times, color='C0', label='OCSMM')
  plt.plot(xs, nocsmm_times, color='orange', linestyle='--', label='N_OCSMM')
  plt.plot(xs, onocsmm_times, color='green', linestyle=':', label='Online_N_OCSMM')

  plt.xlabel('Sample Size')
  plt.ylabel('Computation (seconds)')
  plt.legend()
  
  plt.tight_layout()
  plt.show()
  plt.savefig('./results/plots/sample_size_times.png')

def plot_rolling_window_times(steps, nocsmm_times, onocsmm_times):
  xs = np.array(steps)
  
  plt.figure(figsize=(4, 3.5))
  
  plt.plot(xs, nocsmm_times, color='orange', linestyle='--', label='N_OCSMM')
  plt.plot(xs, onocsmm_times, color='green', linestyle=':', label='Online_N_OCSMM')

  plt.xlabel('Rolling Steps')
  plt.ylabel('Computation (seconds)')
  plt.legend()
  
  plt.tight_layout()
  plt.show()
  plt.savefig('./results/plots/rolling_window_times.png')
