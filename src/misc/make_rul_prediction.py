import numpy as np
import pandas as pd
import torch
import config as cfg
from sksurv.util import Surv
from utility.survival import Survival
from tools.regressors import CoxPH, RSF, DeepSurv, DSM, BNNSurv, CoxBoost, MTLR
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from sklearn.preprocessing import StandardScaler
from utility.survival import make_event_times, make_time_bins
from tools.data_loader import DataLoader
import tensorflow as tf
import random
from sklearn.model_selection import KFold, train_test_split
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

DATASET_NAME = "xjtu"
AXIS = "X"
N_POST_SAMPLES = 100
BEARING_IDS = [1, 2, 3, 4, 5]
K = 2

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

if __name__ == "__main__":
    for condition in [1]: # 0, 1, 2
        dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
        
        for test_bearing_id in BEARING_IDS:
            train_ids = [x for x in BEARING_IDS if x != test_bearing_id]
            
            # Load train data
            train_data = pd.DataFrame()
            for train_bearing_id in train_ids:
                df = dl.make_moving_average(train_bearing_id)
                train_data = pd.concat([train_data, df], axis=0)
            
            # Load test data
            test_data = dl.make_moving_average(test_bearing_id)
            
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
            
            if test_samples.empty:
                test_samples = test_data[test_data['Survival_time'] == test_data['Survival_time'].max()] \
                               .drop_duplicates(subset="Survival_time")
                
            X_train = train_data.drop(['Event', 'Survival_time'], axis=1)
            y_train = Surv.from_dataframe("Event", "Survival_time", train_data)
            X_test = test_samples.drop(['Event', 'Survival_time'], axis=1)
            y_test = Surv.from_dataframe("Event", "Survival_time", test_samples)

            # Set event times for models
            continuous_times = make_event_times(np.array(y_train['Survival_time']), np.array(y_train['Event'])).astype(int)
            continuous_times = np.unique(continuous_times)
            discrete_times = make_time_bins(y_train['Survival_time'].copy(), event=y_train['Event'].copy())
            
            # Scale data
            features = list(X_train.columns)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
            
            # Set up MTLR
            X_train_mtlr, X_valid_mtlr, y_train_mtlr, y_valid_mtlr = train_test_split(X_train_scaled, y_train, test_size=0.3, random_state=0)
            X_train_mtlr = X_train_mtlr.reset_index(drop=True)
            X_valid_mtlr = X_valid_mtlr.reset_index(drop=True)
            data_train = X_train_mtlr.copy()
            data_train["Survival_time"] = pd.Series(y_train_mtlr['Survival_time'])
            data_train["Event"] = pd.Series(y_train_mtlr['Event']).astype(int)
            data_valid = X_valid_mtlr.copy()
            data_valid["Survival_time"] = pd.Series(y_valid_mtlr['Survival_time'])
            data_valid["Event"] = pd.Series(y_valid_mtlr['Event']).astype(int)
            config = dotdict(cfg.PARAMS_MTLR)
            params = MTLR().get_hyperparams(condition)
            config['batch_size'] = params['batch_size']
            config['dropout'] = params['dropout']
            config['lr'] = params['lr']
            config['c1'] = params['c1']
            config['num_epochs'] = params['num_epochs']
            config['hidden_size'] = params['hidden_size']   
            num_features = X_train_mtlr.shape[1]
            num_time_bins = len(discrete_times)
            
            # Set up Cox
            #rsf_model = RSF().make_model(RSF().get_hyperparams(condition))
            
            #Set up BNNSurv
            #bnnsurv_model = BNNSurv().make_model(BNNSurv().get_hyperparams(condition))
            
            # Train models
            mtlr_model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
            mtlr_model = train_mtlr_model(mtlr_model, data_train, data_valid, discrete_times,
                                          config, random_state=0, reset_model=True, device=device)
            """
            mtlr_model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
            mtlr_model = train_mtlr_model(mtlr_model, data_train, data_valid, discrete_times,
                                          config, random_state=0, reset_model=True, device=device)
            bnnsurv_model.fit(X_train_scaled.to_numpy(), y_train['Survival_time'], y_train['Event'])
            rsf_model.fit(X_train_scaled, y_train)
            """
            
            # Predict MTLR
            data_test = X_test_scaled.copy()
            data_test["Survival_time"] = pd.Series(y_test['Survival_time'])
            data_test["Event"] = pd.Series(y_test['Event']).astype(int)
            mtlr_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                          dtype=torch.float, device=device)
            survival_outputs, _, _ = make_mtlr_prediction(mtlr_model, mtlr_test_data, discrete_times, config)
            mtlr_surv_preds = survival_outputs.numpy()
            discrete_times = torch.cat([torch.tensor([0]).to(discrete_times.device), discrete_times], 0)
            mtlr_surv_preds = pd.DataFrame(mtlr_surv_preds, columns=discrete_times.numpy())
            
            # Predict BNNsurv
            #bnnsurv_surv_preds = Survival.predict_survival_function(bnnsurv_model, X_test_scaled, continuous_times, n_post_samples=N_POST_SAMPLES)
            
            # Predict Cox
            #rsf_surv_preds = Survival.predict_survival_function(rsf_model, X_test_scaled, continuous_times)
            
            # Calculate TTE
            for surv_preds in [mtlr_surv_preds]:
                # Sanitize
                surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
                bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
                sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
                sanitized_y_test = np.delete(y_test, bad_idx)
                
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
            print()