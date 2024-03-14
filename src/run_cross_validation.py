import numpy as np
import pandas as pd
import warnings
import config as cfg
from sksurv.util import Surv
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold
from tools.feature_selectors import PHSelector, NoneSelector
from tools.regressors import CoxPH, RSF, DeepSurv, MTLR, BNNSurv, CoxBoost
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from utility.printer import Suppressor
from tools.formatter import Formatter
from tools.evaluator import LifelinesEvaluator
from utility.survival import make_event_times
from tools.cross_validator import run_cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import math
from utility.data import get_window_size, get_lag, get_lmd
from utility.survival import coverage
from scipy.stats._stats_py import chisquare
from utility.event import EventManager
import tensorflow as tf
import random
import torch
from utility.survival import make_time_bins
from utility.mtlr import mtlr, train_mtlr_model, make_mtlr_prediction
from sklearn.model_selection import train_test_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

tf.config.set_visible_devices([], 'GPU') # use CPU

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

NEW_DATASET = False
N_ITER = 10
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_POST_SAMPLES = 100

def main():
    dataset = "xjtu"
    n_boot = 0
    dataset_path = cfg.DATASET_PATH_XJTU
    n_condition = len(cfg.RAW_DATA_PATH_XJTU)
    n_bearing = cfg.N_REAL_BEARING_XJTU
    bearing_ids = list(range(1, (n_bearing*2)+1))

    # For the first time running, a NEW_DATASET is needed
    if NEW_DATASET== True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)

    # Insert the models and feature name selector for CV hyperparameter search and initialize the DataETL instance
    models = [CoxPH, CoxBoost, RSF, MTLR, BNNSurv]
    data_util = DataETL(dataset, n_boot)

    # Extract information from the dataset selected from the config file
    model_results = pd.DataFrame()
    for test_condition in range (0, n_condition):
        timeseries_data, frequency_data = FileReader(dataset, dataset_path).read_data(test_condition, n_boot)
        event_times = EventManager(dataset).get_event_times(frequency_data, test_condition, lmd=get_lmd(test_condition))
        
        # For level of censoring
        for pct in cfg.CENSORING_LEVELS:
            
            # Split in train and test set
            kf = KFold(n_splits=N_OUTER_SPLITS)
            for _, (train_idx, test_idx) in enumerate(kf.split(bearing_ids)):
                # Compute moving average for training/testing
                train_data, test_data = pd.DataFrame(), pd.DataFrame()
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
                
                # Remove frequency and noise features
                unused_features = cfg.FREQUENCY_FTS + cfg.NOISE_FT
                train_data = train_data.drop(unused_features, axis=1)
                test_data = test_data.drop(unused_features, axis=1)
                
                # Add random censoring
                train_data_cens = Formatter.add_random_censoring(train_data, pct=pct)
                test_data_cens = Formatter.add_random_censoring(test_data, pct=pct)
                
                # For all models
                for model_builder in models:
                    model_name = model_builder.__name__
                    
                    # Shuffle train and test data
                    train_data_shuffled = train_data_cens.sample(frac=1, random_state=0)
                    test_data_shuffled = test_data_cens.sample(frac=1, random_state=0)
                    
                    # Format and scale the data
                    train_x = train_data_shuffled.drop(['Event', 'Survival_time'], axis=1)
                    train_y = Surv.from_dataframe("Event", "Survival_time", train_data_shuffled)
                    test_x = test_data_shuffled.drop(['Event', 'Survival_time'], axis=1)
                    test_y = Surv.from_dataframe("Event", "Survival_time", test_data_shuffled)
                    features = list(train_x.columns)
                    scaler = StandardScaler()
                    scaler.fit(train_x)
                    ti = (pd.DataFrame(scaler.transform(train_x), columns=features), train_y)
                    cvi = (pd.DataFrame(scaler.transform(test_x), columns=features), test_y)
                    
                    # Make event times
                    km_times = make_event_times(ti[1]['Survival_time'].copy(), ti[1]['Event'].copy()).astype(int)
                    km_times = np.unique(km_times)
                    mtlr_times = make_time_bins(ti[1]['Survival_time'].copy(), event=ti[1]['Event'].copy())

                    # Find hyperparams via inner CV from hyperparamters' space
                    space = model_builder().get_tuneable_params()
                    param_list = list(ParameterSampler(space, n_iter=N_ITER, random_state=0))
                    best_params = run_cross_validation(model_builder, ti, param_list, device, N_INNER_SPLITS)
                    
                    # Train on train set TI with new parameters
                    x = ti[0].to_numpy()
                    t = ti[1]["Survival_time"]
                    e = ti[1]["Event"]
                    if model_name == "DeepSurv":
                        model = DeepCoxPH(layers=best_params['layers'])
                        with Suppressor():
                            model = model.fit(x, t, e, vsize=0.3, iters=best_params['iters'],
                                                learning_rate=best_params['learning_rate'],
                                                batch_size=best_params['batch_size'])
                    elif model_name == "MTLR":
                        X_train, X_valid, y_train, y_valid = train_test_split(ti[0], ti[1], test_size=0.3, random_state=0)
                        X_train = X_train.reset_index(drop=True)
                        X_valid = X_valid.reset_index(drop=True)
                        data_train = X_train.copy()
                        data_train["Survival_time"] = pd.Series(y_train['Survival_time'])
                        data_train["Event"] = pd.Series(y_train['Event']).astype(int)
                        data_valid = X_valid.copy()
                        data_valid["Survival_time"] = pd.Series(y_valid['Survival_time'])
                        data_valid["Event"] = pd.Series(y_valid['Event']).astype(int)
                        config = dotdict(cfg.PARAMS_MTLR)
                        config['batch_size'] = best_params['batch_size']
                        config['dropout'] = best_params['dropout']
                        config['lr'] = best_params['lr']
                        config['c1'] = best_params['c1']
                        config['num_epochs'] = best_params['num_epochs']
                        config['hidden_size'] = best_params['hidden_size']
                        num_features = ti[0].shape[1]
                        num_time_bins = len(mtlr_times)
                        model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
                        model = train_mtlr_model(model, data_train, data_valid, mtlr_times,
                                                 config, random_state=0, reset_model=True, device=device)
                    elif model_name == "DSM":
                        model = DeepSurvivalMachines(layers=best_params['layers'])
                        with Suppressor():
                            model = model.fit(x, t, e, vsize=0.3, iters=best_params['iters'],
                                                learning_rate=best_params['learning_rate'],
                                                batch_size=best_params['batch_size'])
                    elif model_name == "BNNmcd":
                        model = model_builder().make_model(best_params)
                        with Suppressor():
                            model.fit(x, t, e)
                    else:
                        model = model_builder().make_model(best_params)
                        with Suppressor():
                            model.fit(ti[0], ti[1])
                    
                    # Get survival predictions for CVI
                    if model_name == "DeepSurv" or model_name == "DSM" or model_name == "BNNmcd":
                        xte = cvi[0].to_numpy()
                        surv_preds = Survival.predict_survival_function(model, xte, km_times, n_post_samples=N_POST_SAMPLES)
                    elif model_name == "MTLR":
                        data_test = cvi[0].copy()
                        data_test["Survival_time"] = pd.Series(cvi[1]['Survival_time'])
                        data_test["Event"] = pd.Series(cvi[1]['Event']).astype(int)
                        baycox_test_data = torch.tensor(data_test.drop(["Survival_time", "Event"], axis=1).values,
                                                        dtype=torch.float, device=device)
                        survival_outputs, _, _ = make_mtlr_prediction(model, baycox_test_data, mtlr_times, config)
                        surv_preds = survival_outputs.numpy()
                        mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
                        surv_preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
                    else:
                        surv_preds = Survival.predict_survival_function(model, cvi[0], km_times)
                        
                    # Sanitize
                    surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
                    bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
                    sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
                    sanitized_cvi = np.delete(cvi[1], bad_idx)
                    
                    # Calculate scores
                    try:
                        lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_cvi['Survival_time'], sanitized_cvi['Event'],
                                                            ti[1]['Survival_time'], ti[1]['Event'])
                        mae_hinge = lifelines_eval.mae(method="Hinge")
                        mae_margin = lifelines_eval.mae(method="Margin")
                        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                        d_calib = lifelines_eval.d_calibration()[0]
                    except:
                        mae_hinge = np.nan
                        mae_margin = np.nan
                        mae_pseudo = np.nan
                        d_calib = np.nan

                    if mae_hinge > 500:
                        mae_hinge = np.nan
                    if mae_margin > 500:
                        mae_margin = np.nan
                    if mae_pseudo > 500:
                        mae_pseudo = np.nan
                    
                    if test_condition == 0:
                        cond_name = "C1"
                    elif test_condition == 1:
                        cond_name = "C2"
                    else:
                        cond_name = "C3"
                    
                    # Calucate C-cal for BNN model
                    if model_name == "BNNmcd":
                        xte = cvi[0].to_numpy()
                        surv_probs = model.predict_survival(xte, event_times=km_times, n_post_samples=N_POST_SAMPLES)
                        credible_region_sizes = np.arange(0.1, 1, 0.1)
                        surv_times = torch.from_numpy(surv_probs)
                        coverage_stats = {}
                        for percentage in credible_region_sizes:
                            drop_num = math.floor(0.5 * N_POST_SAMPLES * (1 - percentage))
                            lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                            upper_outputs = torch.kthvalue(surv_times, k=N_POST_SAMPLES - drop_num, dim=0)[0]
                            coverage_stats[percentage] = coverage(km_times, upper_outputs, lower_outputs,
                                                                    cvi[1]["Survival_time"], cvi[1]["Event"])
                        data = [list(coverage_stats.keys()), list(coverage_stats.values())]
                        _, pvalue = chisquare(data)
                        c_calib = pvalue[0]
                    else:
                        c_calib = 0
                        
                    print(f"Evaluated {cond_name} - {model_name} - {pct}")
                    res_sr = pd.Series([cond_name, model_name, pct, best_params,
                                        mae_hinge, mae_margin, mae_pseudo, d_calib, c_calib],
                                        index=["Condition", "ModelName", "CensoringLevel", "BestParams",
                                               "MAEHinge", "MAEMargin", "MAEPseudo", "DCalib", "CCalib"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                    model_results.to_csv(f"{cfg.RESULTS_PATH}/model_results.csv")

if __name__ == "__main__":
    main()