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
from tools.regressors import CoxPH, CoxBoost, RSF, MTLR, BNNSurv
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
n_bearing = cfg.N_REAL_BEARING_XJTU
bearing_ids = cfg.BEARING_IDS
pct_censoring = 0.25
n_post_samples = 100

if __name__ == "__main__":
    data_util = DataETL(dataset, n_boot)
    event_manager = EventManager(dataset)
    
    if new_dataset == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in [0, 1, 2]:
        timeseries_data, frequency_data = FileReader(dataset, dataset_path).read_data(test_condition, axis=axis)
        event_times = EventManager(dataset).get_event_times(frequency_data, test_condition, lmd=get_lmd(test_condition))
        train_data, test_data = pd.DataFrame(), pd.DataFrame()
        train_ids = [1, 2, 3] # Bearings 1-3
        test_ids = [4, 5] # Bearing 4-5
        for train_bearing_id in train_ids:
            event_time = event_times[train_bearing_id-1]
            transformed_data = data_util.make_moving_average(timeseries_data, event_time, train_bearing_id,
                                                             get_window_size(test_condition),
                                                             get_lag(test_condition))
            train_data = pd.concat([train_data, transformed_data], axis=0)
            train_data = train_data.reset_index(drop=True)
        for test_bearing_id in test_ids:
            event_time = event_times[test_bearing_id-1]
            transformed_data = data_util.make_moving_average(timeseries_data, event_time, test_bearing_id,
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
        
        # Make discrete times
        discrete_times = make_time_bins(y_train['Survival_time'].copy(), event=y_train['Event'].copy())
        
        # Scale data
        features = list(X_train.columns)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

        # Format training data for BNN model
        X_train_arr = X_train.to_numpy()
        X_test_arr = X_test.to_numpy()
        t_train = y_train['Survival_time']
        e_train = y_train['Event']
        
        # Format data for MTLR model
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
        X_train = X_train.reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        data_train = X_train.copy()
        data_train["Survival_time"] = pd.Series(y_train['Survival_time'])
        data_train["Event"] = pd.Series(y_train['Event']).astype(int)
        data_valid = X_valid.copy()
        data_valid["Survival_time"] = pd.Series(y_valid['Survival_time'])
        data_valid["Event"] = pd.Series(y_valid['Event']).astype(int)
        
        # Make models
        cph_model = CoxPH().make_model(CoxPH().get_best_hyperparams(test_condition))
        coxboost_model = CoxBoost().make_model(CoxBoost().get_best_hyperparams(test_condition))
        rsf_model = RSF().make_model(RSF().get_best_hyperparams(test_condition))
        bnn_model = BNNSurv().make_model(BNNSurv().get_best_hyperparams(test_condition))
        config = dotdict(cfg.PARAMS_MTLR)
        best_params = MTLR().get_best_hyperparams(test_condition)
        config['batch_size'] = best_params['batch_size']
        config['dropout'] = best_params['dropout']
        config['lr'] = best_params['lr']
        config['c1'] = best_params['c1']
        config['num_epochs'] = best_params['num_epochs']
        config['hidden_size'] = best_params['hidden_size']
        num_features = X_train.shape[1]
        num_time_bins = len(discrete_times)
        mtlr_model = MTLR().make_model(num_features=num_features,
                                       num_time_bins=num_time_bins,
                                       config=config)
    
        # Train models
        cph_model.fit(X_train, y_train)
        coxboost_model.fit(X_train, y_train)
        rsf_model.fit(X_train, y_train)
        bnn_model.fit(X_train_arr, t_train, e_train)
        mtlr_model = train_mtlr_model(mtlr_model, data_train, data_valid, discrete_times,
                                      config, random_state=0, reset_model=True, device=device)
        #Predict
        unique_times = make_event_times(y_train['Survival_time'].copy(), y_train['Event'].copy()).astype(int)
        unique_times = np.unique(unique_times)
        km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Survival_time'].reshape(1,-1),
                                                               y_test['Event'].reshape(1,-1),
                                                               unique_times)
        cph_surv_func = Survival().predict_survival_function(cph_model, X_test, unique_times)
        coxboost_surv_func = Survival().predict_survival_function(coxboost_model, X_test, unique_times)
        rsf_surv_func = Survival().predict_survival_function(rsf_model, X_test, unique_times)
        bnn_surv_func = Survival().predict_survival_function(bnn_model, X_test_arr, unique_times,
                                                             n_post_samples=n_post_samples)
        data_test = X_test.copy()
        data_test["Survival_time"] = pd.Series(y_test['Survival_time'])
        data_test["Event"] = pd.Series(y_test['Event']).astype(int)
        baycox_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                        dtype=torch.float, device=device)
        survival_outputs, _, _ = make_mtlr_prediction(mtlr_model, baycox_test_data, discrete_times, config)
        surv_preds = survival_outputs.numpy()
        discrete_times = torch.cat([torch.tensor([0]).to(discrete_times.device), discrete_times], 0)
        mtlr_surv_func = pd.DataFrame(surv_preds, columns=discrete_times.numpy())

        fig = plt.figure(figsize=(6, 4))
        plt.plot(km_mean.columns, km_mean.iloc[0], 'k--', linewidth=2, alpha=0.5, label=r"$\mathbb{E}[S(t)]$ Kaplan-Meier", color="black")
        plt.plot(unique_times, np.mean(cph_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ CoxPH", linewidth=2)
        plt.plot(unique_times, np.mean(coxboost_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ CoxBoost", linewidth=2)
        plt.plot(unique_times, np.mean(rsf_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ RSF", linewidth=2)
        plt.plot(discrete_times, np.mean(mtlr_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ MTLR", linewidth=2)
        plt.plot(unique_times, np.mean(bnn_surv_func, axis=0), label=r"$\mathbb{E}[S(t|\bm{X})]$ BNNSurv", linewidth=2)
        plt.xlabel("Time (min)")
        plt.ylabel("Survival probability S(t)")
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.savefig(f'{cfg.PLOTS_PATH}/mean_survival_C{test_condition+1}.pdf', format='pdf', bbox_inches="tight")
        