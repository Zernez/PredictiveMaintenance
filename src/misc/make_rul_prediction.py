import numpy as np
import pandas as pd
import math
import torch
import config as cfg
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.regressors import CoxPH, RSF, DeepSurv, DSM, BNNSurv, CoxBoost, MTLR
from tools.feature_selectors import PHSelector
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines.statistics import proportional_hazard_test
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time, predict_mean_survival_time
import config as cfg
from sklearn.preprocessing import StandardScaler
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times
from utility.data import get_window_size, get_lag
import os
import contextlib
from utility.event import EventManager
from utility.data import get_window_size, get_lag, get_lmd
from tools.data_loader import DataLoader
import tensorflow as tf
import random

np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

DATASET_NAME = "xjtu"
AXIS = "X"
N_POST_SAMPLES = 100
BEARING_IDS = [1, 2, 3, 4, 5]
K = 1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == "__main__":
    for condition in [0]: # 0, 1, 2
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        
        for test_bearing_id in BEARING_IDS:
            train_ids = [x for x in BEARING_IDS if x != test_bearing_id]
            
            # Load train data
            train_data = pd.DataFrame()
            for train_bearing_id in train_ids:
                df = dl.make_moving_average(train_bearing_id, drop_non_ph_fts=False)
                train_data = pd.concat([train_data, df], axis=0)
            
            # Load test data
            test_data = dl.make_moving_average(test_bearing_id, drop_non_ph_fts=False)
            
            # Reset index
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            # Select Tk observations
            surv_times = range(1, int(test_data['Survival_time'].max()+1))
            test_samples = pd.DataFrame()
            for k in range(1, K+1):
                tk = int(np.quantile(surv_times, 1 / k))
                tk_nearest = find_nearest(surv_times, tk)
                test_sample = test_data[test_data['Survival_time'] == tk_nearest]
                test_samples = pd.concat([test_samples, test_sample], axis=0)
            test_samples = test_samples.loc[test_samples['Event'] == True]
                
            X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            X_test = test_samples.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_samples)

            # Set event times for models
            continuous_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            continuous_times = np.unique(continuous_times)

            # Scale data
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #Set up the models
            model = BNNSurv().make_model(BNNSurv().get_best_hyperparams(condition))
            #model = CoxBoost().make_model(CoxBoost().get_best_hyperparams(condition))
            
            # Train the model
            model.fit(X_train_scaled, y_train['Survival_time'], y_train['Event'])
            #model.fit(X_train_scaled, y_train)
            
            # Predict
            surv_preds = Survival.predict_survival_function(model, X_test_scaled, continuous_times, n_post_samples=N_POST_SAMPLES)

            # Sanitize
            surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
            bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
            sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
            sanitized_y_test = np.delete(y_test, bad_idx)
            
            # Calculate TTE
            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_y_test['Survival_time'], sanitized_y_test['Event'],
                                                y_train['Survival_time'], y_train['Event'])
            median_survs = lifelines_eval.predict_time_from_curve(predict_median_survival_time)
                    
            # Calculate CRA
            cra = 0
            n_preds = len(median_survs)
            for k in range(1, n_preds+1):
                wk = k/sum(range(n_preds+1))
                ra_tk = 1 - (abs(sanitized_y_test['Survival_time'][k-1]-median_survs[k-1])/
                             sanitized_y_test['Survival_time'][k-1])
                cra += wk*ra_tk
            print(f'CRA for Bearing {condition+1}_{test_bearing_id}: {round(cra, 4)}')