import numpy as np
import pandas as pd
from time import time
import warnings
import config as cfg
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold
from tools.feature_selectors import PHSelector
from tools.regressors import CoxPH, RSF, DeepSurv, DSM, BNNmcd
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from utility.printer import Suppressor
from tools.formatter import Formatter
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from utility.survival import make_event_times
from tools.cross_validator import run_cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import argparse
import torch
import math
from utility.data import get_window_size, get_lag, get_lmd
from utility.survival import coverage
from scipy.stats._stats_py import chisquare
from utility.event import EventManager

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)

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
    
    # For the first time running, a NEW_DATASET is needed
    if NEW_DATASET== True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)

    # Insert the models and feature name selector for CV hyperparameter search and initialize the DataETL instance
    models = [CoxPH, RSF, DeepSurv, DSM, BNNmcd]
    ft_selectors = [PHSelector]
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
            bearing_indicies = list(range(1, (n_bearing*2)+1)) # number of real bearings x 2
            for _, (train_idx, test_idx) in enumerate(kf.split(bearing_indicies)):
                # Track time
                split_start_time = time()
                
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
                
                # Add random censoring
                train_data_cens = Formatter.add_random_censoring(train_data, pct=pct)
                test_data_cens = Formatter.add_random_censoring(test_data, pct=pct)
                
                # For all models
                for model_builder in models:
                    model_name = model_builder.__name__
                    
                    # For all feature selectors
                    for ft_selector_builder in ft_selectors:
                        ft_selector_name = ft_selector_builder.__name__
                                                
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
                        
                        # Create model instance and find best features
                        model = model_builder().get_estimator()
                        if ft_selector_name == "PHSelector":
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=[dataset, test_condition])
                        else:
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                        
                        # Format the data with the feature selected
                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        ti_new[0].reset_index(inplace=True, drop=True)
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        cvi_new[0].reset_index(inplace=True, drop=True)
                        
                        # Make event times
                        times = make_event_times(ti_new[1]['Survival_time'].copy(), ti_new[1]['Event'].copy()).astype(int)
                        times = np.unique(times)

                        # Find hyperparams via inner CV from hyperparamters' space
                        space = model_builder().get_tuneable_params()
                        param_list = list(ParameterSampler(space, n_iter=N_ITER, random_state=0))
                        best_params = run_cross_validation(model_builder, ti_new, param_list, N_INNER_SPLITS)
                        
                        # Train on train set TI with new parameters
                        x = ti_new[0].to_numpy()
                        t = ti_new[1]["Survival_time"]
                        e = ti_new[1]["Event"]
                        if model_name == "DeepSurv":
                            model = DeepCoxPH(layers=best_params['layers'])
                            with Suppressor():
                                model = model.fit(x, t, e, vsize=0.3, iters=best_params['iters'],
                                                    learning_rate=best_params['learning_rate'],
                                                    batch_size=best_params['batch_size'])
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
                                model.fit(ti_new[0], ti_new[1])
                        
                        # Get survival predictions for CVI
                        if model_name == "DeepSurv" or model_name == "DSM" or model_name == "BNNmcd":
                            xte = cvi_new[0].to_numpy()
                            with Suppressor():
                                surv_preds = Survival.predict_survival_function(model, xte, times, n_post_samples=N_POST_SAMPLES)
                        else:
                            with Suppressor():
                                surv_preds = Survival.predict_survival_function(model, cvi_new[0], times)
                            
                        # Sanitize
                        surv_preds = surv_preds.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0.001)
                        bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index # check we have a median
                        sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
                        sanitized_cvi = np.delete(cvi_new[1], bad_idx)
                        
                        # Calculate scores
                        try:
                            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_cvi['Survival_time'], sanitized_cvi['Event'],
                                                                ti_new[1]['Survival_time'], ti_new[1]['Event'])
                            mae_hinge = lifelines_eval.mae(method="Hinge")
                            mae_margin = lifelines_eval.mae(method="Margin")
                            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
                            d_calib = 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0
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
                        
                        n_preds = len(surv_preds)
                        t_total_split_time = time() - split_start_time

                        if test_condition == 0:
                            cond_name = "C1"
                        elif test_condition == 1:
                            cond_name = "C2"
                        else:
                            cond_name = "C3"
                        
                        # Calucate C-cal for BNN model
                        if model_name == "BNNmcd":
                            surv_probs = model.predict_survival(xte, event_times=times, n_post_samples=N_POST_SAMPLES)
                            credible_region_sizes = np.arange(0.1, 1, 0.1)
                            surv_times = torch.from_numpy(surv_probs)
                            coverage_stats = {}
                            for percentage in credible_region_sizes:
                                drop_num = math.floor(0.5 * N_POST_SAMPLES * (1 - percentage))
                                lower_outputs = torch.kthvalue(surv_times, k=1 + drop_num, dim=0)[0]
                                upper_outputs = torch.kthvalue(surv_times, k=N_POST_SAMPLES - drop_num, dim=0)[0]
                                coverage_stats[percentage] = coverage(times, upper_outputs, lower_outputs,
                                                                        cvi_new[1]["Survival_time"], cvi_new[1]["Event"])
                            data = [list(coverage_stats.keys()), list(coverage_stats.values())]
                            _, pvalue = chisquare(data)
                            alpha = 0.05
                            if pvalue[0] <= alpha:
                                c_calib = 0
                            else:
                                c_calib = 1
                        else:
                            c_calib = 0
                            
                        print(f"Evaluated {cond_name} - {model_name} - {ft_selector_name} - {pct}")
                        res_sr = pd.Series([cond_name, model_name, ft_selector_name, pct,
                                            mae_hinge, mae_margin, mae_pseudo, d_calib, c_calib,
                                            n_preds, t_total_split_time, best_params, list(selected_fts)],
                                            index=["Condition", "ModelName", "FtSelectorName", "CensoringLevel",
                                                   "MAEHinge", "MAEMargin", "MAEPseudo", "DCalib", "CCalib",
                                                   "Npreds", "TTotalSplit", "BestParams", "SelectedFts"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                        model_results.to_csv(f"{cfg.RESULTS_PATH}/model_results.csv")

if __name__ == "__main__":
    main()