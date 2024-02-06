import numpy as np
import pandas as pd
import math
import torch
import config as cfg
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.regressors import CoxPH, RSF, DeepSurv, DSM, BNNmcd
from tools.feature_selectors import PHSelector
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines.statistics import proportional_hazard_test
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
import config as cfg
from sklearn.preprocessing import StandardScaler
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times
from utility.data import get_window_size, get_lag
import os
import contextlib

NEW_DATASET = False
DATASET = "xjtu"
N_BOOT = 0
N_POST_SAMPLES = 1000
DATASET_PATH = cfg.DATASET_PATH_XJTU
K = 1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == "__main__":
    if NEW_DATASET== True:
        Builder(DATASET, N_BOOT).build_new_dataset(bootstrap=N_BOOT)
        
    data_util = DataETL(DATASET, N_BOOT)
    survival = Survival()
    x_bearings = [idx for idx in list(range(1, 11)) if idx % 2 != 0]

    for cond in [0, 1, 2]:
        # Build timeseries data
        timeseries_data, boot, info_pack = FileReader(DATASET, DATASET_PATH).read_data(cond, N_BOOT)
        window_size = get_window_size(cond)
        lag = get_lag(cond)
        
        # Individual bearing prediction
        bearings = x_bearings
        print_idx = list(range(1, 6))
        for test_idx, print_idx in zip(bearings, print_idx):
            train_idx = [x for x in bearings if x != test_idx]
            
            train_data = pd.DataFrame()
            for idx in train_idx:
                event_time = data_util.event_analyzer(idx, info_pack)
                transformed_data = data_util.make_moving_average(timeseries_data, event_time, idx, window_size, lag)
                train_data = pd.concat([train_data, transformed_data], axis=0)
                
            test_event_time = data_util.event_analyzer(test_idx, info_pack)
            test_data = data_util.make_moving_average(timeseries_data, test_event_time, test_idx, window_size, lag)
                    
            train_data = Formatter.add_random_censoring(train_data, percentage=0.25)
            test_data = Formatter.add_random_censoring(test_data, percentage=0.25)
            train_data = train_data.sample(frac=1, random_state=0)
            test_data = test_data.sample(frac=1, random_state=0)
            
            # Select Tk observations
            surv_times = range(1, int(test_data['Survival_time'].max()))
            test_samples = pd.DataFrame()
            for k in range(1, K+1):
                tk = int(np.quantile(surv_times, 1 / k))
                surv_times = list(test_data['Survival_time'])
                tk_nearest = find_nearest(surv_times, tk)
                test_sample = test_data[test_data['Survival_time'] == tk_nearest]
                test_samples = pd.concat([test_samples, test_sample], axis=0)
            test_samples = test_samples.loc[test_samples['Event'] == True]
                
            x_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            x_test = test_data.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_data)

            #Set event times for models
            event_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            event_times = np.unique(event_times)

            #Set the feature selector and train/test split
            best_features = PHSelector(x_train, y_train, estimator=[DATASET, cond]).get_features()
            X_train, X_test = x_train.loc[:,best_features], x_test.loc[:,best_features]

            # Scale data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #Set up the models
            model = BNNmcd().make_model(BNNmcd().get_best_hyperparams())
            #model = DeepSurv().make_model(DeepSurv().get_best_hyperparams())
            #model = DSM().make_model(DSM().get_best_hyperparams())
            #model = CoxPH().make_model(CoxPH().get_best_hyperparams())

            # Train the model
            #model.fit(X_train_scaled, y_train['Survival_time'], y_train['Event'])
            #params = DeepSurv().get_best_hyperparams()
            #params = DSM().get_best_hyperparams()
            #model = model.fit(X_train_scaled, y_train['Survival_time'], y_train['Event'],
            #                  vsize=0.3, iters=params['iters'],
            #                  learning_rate=params['learning_rate'],
            #                  batch_size=params['batch_size'])
            model.fit(X_train_scaled, y_train)
            
            # Predict
            surv_preds = survival.predict_survival_function(model, X_test_scaled, event_times, n_post_samples=N_POST_SAMPLES)
            
            # Sanitize
            surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
            bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
            sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
            sanitized_y_test = np.delete(y_test, bad_idx)
            
            # Calculate TTE
            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test['Survival_time'],
                                                sanitized_y_test['Event'], y_train['Survival_time'], y_train['Event'])
            median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
            
            # Calculate CRA
            cra = 0
            n_preds = len(median_survs)
            for k in range(1, n_preds+1):
                wk = k/sum(range(n_preds+1))
                ra_tk = 1 - (abs(sanitized_y_test['Survival_time'][k-1]-median_survs[k-1])/
                             sanitized_y_test['Survival_time'][k-1])
                cra += wk*ra_tk
            print(f'CRA for Bearing_{cond+1}_{print_idx}: {round(cra, 4)}')