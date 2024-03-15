import numpy as np
import pandas as pd
import config as cfg
from sksurv.util import Surv
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times, make_time_bins
from utility.data import get_window_size, get_lag, get_lmd
from utility.event import EventManager
from sklearn.preprocessing import StandardScaler
from tools.regressors import BNNSurv
from utility.survival import Survival
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
import torch
from sklearn.model_selection import train_test_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
                     'font.size': 12.0,
                     'text.usetex': True,
                     'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

new_dataset = False
dataset = "xjtu"
axis = "X"
n_boot = 0
dataset_path = cfg.DATASET_PATH_XJTU
pct_censoring = 0.25
n_post_samples = 100

if __name__ == "__main__":
    data_util = DataETL(dataset, n_boot)
    event_manager = EventManager(dataset)
    
    if new_dataset == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in [1]: # use C1
        timeseries_data, frequency_data = FileReader(dataset, dataset_path).read_data(test_condition, axis=axis)
        event_times = EventManager(dataset).get_event_times(frequency_data, test_condition, lmd=get_lmd(test_condition))
        train_data, test_data = pd.DataFrame(), pd.DataFrame()
        train_idx = [0, 1, 2, 3] # Bearings 1-4
        test_idx = [4] # Bearing 5
        for idx in train_idx:
            event_time = event_times[idx]
            transformed_data = data_util.make_moving_average(timeseries_data, event_time, idx+1,
                                                             get_window_size(test_condition),
                                                             get_lag(test_condition))
            train_data = pd.concat([train_data, transformed_data], axis=0)
            train_data = train_data.reset_index(drop=True)
        for idx in test_idx:
            event_time = event_times[idx]
            transformed_data = data_util.make_moving_average(timeseries_data, event_time, idx+1,
                                                             get_window_size(test_condition),
                                                             get_lag(test_condition))
            test_data = pd.concat([test_data, transformed_data], axis=0)
            test_data = test_data.reset_index(drop=True)
        
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        unused_features = cfg.FREQUENCY_FTS + cfg.NOISE_FT
        train_data = train_data.drop(unused_features, axis=1)
        test_data = test_data.drop(unused_features, axis=1)
        train_data = Formatter.add_random_censoring(train_data, pct=pct_censoring)
        test_data = Formatter.add_random_censoring(test_data, pct=pct_censoring)
        train_data = train_data.sample(frac=1, random_state=0)
        test_data = test_data.sample(frac=1, random_state=0)
        
        X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
        y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
        X_test = test_data.drop(['Event', 'Survival_time'], axis=1)
        y_test = Surv.from_dataframe("Event", "Survival_time", test_data)
        
        # Make unique times
        unique_times = make_event_times(y_train['Survival_time'].copy(), y_train['Event'].copy()).astype(int)
        unique_times = np.unique(unique_times)
    
        #Format the data
        t_train = y_train['Survival_time']
        e_train = y_train['Event']
        t_test = y_test['Survival_time']
        e_test = y_test['Event']

        # Predict per feature
        features_to_split = X_train.columns
        for feature in features_to_split:
            X_train_feature, X_test_feature = pd.DataFrame(X_train[feature]), pd.DataFrame(X_test[feature])
        
            # Scale train data
            scaler = StandardScaler()
            scaler.fit(X_train_feature)
            X_train_scaled = scaler.transform(X_train_feature)
            
            model = BNNSurv().make_model(BNNSurv().get_best_hyperparams())
            model.fit(X_train_scaled, t_train, e_train)
            
            # Split data
            split_thresholds = []
            for qct in [0.25, 0.5, 0.75]:
                split_thresholds.append(round(X_test_feature[feature].quantile(qct), 2))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
            for i, st in enumerate(split_thresholds):
                g1_idx = X_test_feature.loc[X_test_feature[feature] < st].index
                g2_idx = X_test_feature.loc[X_test_feature[feature] >= st].index
                X_test_g1 = X_test_feature.loc[g1_idx]
                X_test_g2 = X_test_feature.loc[g2_idx]
                y_test_g1 = y_test[g1_idx]
                y_test_g2 = y_test[g2_idx]
        
                # Scale splitted data
                X_test_scaled = scaler.transform(X_test_feature)
                X_test_g1_scaled = scaler.transform(X_test_g1)
                X_test_g2_scaled = scaler.transform(X_test_g2)
        
                # Predict for mean and two groups
                surv_probs = Survival().predict_survival_function(model, X_test_scaled, unique_times,
                                                                  n_post_samples=n_post_samples)
                km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Survival_time'].reshape(1,-1),
                                                                    y_test['Event'].reshape(1,-1),
                                                                    unique_times)
                surv_probs_g1 = Survival().predict_survival_function(model, X_test_g1_scaled, unique_times,
                                                                     n_post_samples=n_post_samples)
                km_mean_g1, km_high_g1, km_low_g1 = calculate_kaplan_vectorized(y_test_g1['Survival_time'].reshape(1,-1),
                                                                                y_test_g1['Event'].reshape(1,-1),
                                                                                unique_times)
                surv_probs_g2 = Survival().predict_survival_function(model, X_test_g2_scaled, unique_times,
                                                                     n_post_samples=n_post_samples)
                
                surv_probs_mean = np.mean(surv_probs, axis=0)
                surv_probs_g1_mean = np.mean(surv_probs_g1, axis=0)
                surv_probs_g2_mean = np.mean(surv_probs_g2, axis=0)
                
                # Plot
                axes[i].plot(surv_probs_mean, linewidth=2, label=r"$\mathbb{E}[S(t|$" + f"{feature}" + r"$)]$", color="black")
                axes[i].plot(surv_probs_g1_mean, linewidth=2, label=r"$S(t|$" + f"{feature} $<{st})$", color="C0")
                axes[i].plot(surv_probs_g2_mean, linewidth=2, label=r"$S(t|$" + f"{feature} $\geq{st})$", color="C1")
                axes[i].set_xlabel("Time (min)")
                axes[i].set_ylabel("Survival probability S(t)")
                axes[i].legend(loc='upper right')
                axes[i].grid(True)
            plt.savefig(f'{cfg.PLOTS_PATH}/group_survival_{feature}_C{test_condition+1}.pdf',
                        format='pdf', bbox_inches="tight")
        