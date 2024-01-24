import numpy as np
import pandas as pd
from time import time
import warnings
import config as cfg
import re
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import KFold
from tools.feature_selectors import PHSelector, NoneSelector
from tools.regressors import CoxPH, CphLASSO, RSF, DeepSurv, DSM, BNNmcd
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
from tools.Evaluations.TargetRUL import estimate_target_rul_xjtu
from tools.Evaluations.TargetRUL import estimate_target_rul_pronostia
from utility.survival import make_event_times
from tools.cross_validator import run_cross_validation
from xgbse.metrics import approx_brier_score
import os
import argparse

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

NEW_DATASET = True
N_INTERNAL_SPLITS = 5
N_ITER = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    parser.add_argument('--typedata', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    
    global DATASET
    global TYPE
    global N_CONDITION
    global N_BEARING
    global N_SPLITS
    global TRAIN_SIZE
    global CENSORING
    global N_BOOT
    
    if args.dataset:
        DATASET = args.dataset
        cfg.DATASET_NAME = args.dataset

    if args.typedata:
        TYPE = args.typedata
    
    # DATASET= "xjtu"
    # TYPE= "bootstrap"

    if TYPE == "bootstrap":
        N_BOOT = 16
    else:
        N_BOOT = 0

    if DATASET == "xjtu":
        data_path = cfg.RAW_DATA_PATH_XJTU
        N_CONDITION = len(data_path)
        N_BEARING = cfg.N_REAL_BEARING_XJTU
        N_SPLITS = 5
        TRAIN_SIZE = 1
        CENSORING = cfg.CENSORING_LEVEL
    elif DATASET == "pronostia":
        data_path = cfg.RAW_DATA_PATH_PRONOSTIA
        N_CONDITION = len(data_path)
        N_BEARING = cfg.N_REAL_BEARING_PRONOSTIA
        N_SPLITS = 2
        TRAIN_SIZE = 1
        CENSORING = cfg.CENSORING_LEVEL
    
    # For the first time running, a NEW_DATASET is needed
    if NEW_DATASET== True:
        Builder(DATASET, N_BOOT, TYPE).build_new_dataset(bootstrap=N_BOOT)

    # Insert the models and feature name selector for CV hyperparameter search and initialize the DataETL instance
    models = [CoxPH, RSF, DeepSurv, DSM, BNNmcd]
    ft_selectors = [PHSelector]
    data_util = DataETL(DATASET, N_BOOT)

    # Extract information from the dataset selected from the config file
    cov_group = []
    boot_group = []
    info_group = []
    for test_condition in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET, TYPE).read_data(test_condition, N_BOOT)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    # Transform information from the dataset selected from the config file
    data_container_X = []
    data_container_y= []
    for test_condition, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
        # Create different data for bootstrap and not bootstrap
        if TYPE == "bootstrap":
            data_temp_X, deltaref_temp_y = data_util.make_surv_data_bootstrap(cov, boot, info_pack, N_BOOT)
        elif TYPE == "not_correlated":
            data_temp_X, deltaref_temp_y = data_util.make_surv_data_transform_ma(cov, boot, info_pack, N_BOOT, TYPE)
        elif TYPE == "correlated":
            data_temp_X, deltaref_temp_y = data_util.make_surv_data_transform_ama(cov, boot, info_pack, N_BOOT, TYPE)
        data_container_X.append(data_temp_X)
        data_container_y.append(deltaref_temp_y)

    # Load information from the dataset selected in the config file                                                                          
    for test_condition, (data, data_y) in enumerate(zip(data_container_X, data_container_y)):

        # Information about the event estimation in event detector
        y_delta = data_y

        # Iteration for each censored condition
        for censor_condition, percentage in enumerate(CENSORING):

            # Eventually control the censored data by CENSORING
            data_X= []
            for data_group in data:   
                data_temp_X = Formatter.control_censored_data(data_group, percentage= percentage)
                data_X.append(data_temp_X)

            # Indexing by the original bearing number the dataset to avoid train/test leaking. 
            # The data will be splitted in chunks of bearings bootrastrapped from the original bearing.
            dummy_x = list(range(0, int(np.floor(N_BEARING * TRAIN_SIZE)), 1))
            
            print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors. Dataset: {DATASET}. Type: {TYPE}")
            
            # For all models selected
            for model_builder in models:
                model_name = model_builder.__name__
                model_results = pd.DataFrame()

                # For all feature selector selected
                for ft_selector_builder in ft_selectors:
                    ft_selector_name = ft_selector_builder.__name__
                    
                    # For N_SPLITS folds 
                    kf = KFold(n_splits= N_SPLITS, shuffle= False)
                    for split_idx, (train, test) in enumerate(kf.split(dummy_x)):
                        # Start take the time for search the hyperparameters for each fold
                        split_start_time = time()

                        # Load the train data from group indexed Kfold splitting avoiding train/test leaking                
                        data_X_merge = pd.DataFrame()
                        for element in train:
                            data_X_merge = pd.concat([data_X_merge, data_X [element]], ignore_index=True)
                        data_y = Surv.from_dataframe("Event", "Survival_time", data_X_merge)
                        T1 = (data_X_merge, data_y)

                        # Load the test data from group indexed Kfold splitting avoiding train/test leaking               
                        data_X_merge = pd.DataFrame()
                        for element in test:
                            data_X_merge = pd.concat([data_X_merge, data_X [element]], ignore_index=True)
                        data_y = Surv.from_dataframe("Event", "Survival_time", data_X_merge)
                        T2 = (data_X_merge, data_y)

                        # Fromat and center the data
                        ti, cvi, ti_NN, cvi_NN = Formatter.format_main_data(T1, T2)
                        ti, cvi, ti_NN, cvi_NN = Formatter.centering_main_data(ti, cvi, ti_NN, cvi_NN)

                        ft_selector_print_name = f"{ft_selector_name}"
                        model_print_name = f"{model_name}"
                        
                        # Create model instance and find best features
                        model = model_builder().get_estimator()
                        if ft_selector_name == "PHSelector":
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=[DATASET, TYPE])
                        else:
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                        
                        # Format the data with the feature selected
                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        ti_new[0].reset_index(inplace=True, drop=True)
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        cvi_new[0].reset_index(inplace=True, drop=True)
                        ti_new_NN =  (ti_NN[0].loc[:, selected_fts], ti_NN[1])
                        ti_new_NN[0].reset_index(inplace=True, drop=True)
                        cvi_new_NN = (cvi_NN[0].loc[:, selected_fts], cvi_NN[1])
                        cvi_new_NN[0].reset_index(inplace=True, drop=True)

                        # Set event times
                        times = make_event_times(ti_new_NN[1]['time'], ti_new_NN[1]['event']).astype(int)
                        times = np.unique(times)

                        # Find hyperparams via inner CV from hyperparamters' space
                        space = model_builder().get_tuneable_params()
                        param_list = list(ParameterSampler(space, n_iter=N_ITER, random_state=0))
                        best_params = run_cross_validation(model_builder, ti_new_NN,
                                                           param_list, N_INTERNAL_SPLITS)
                        
                        # Train on train set TI with new parameters
                        x = ti_new_NN[0].to_numpy()
                        t = ti_new_NN[1].loc[:, "time"].to_numpy()
                        e = ti_new_NN[1].loc[:, "event"].to_numpy()
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
                            xte = cvi_new_NN[0].to_numpy()
                            with Suppressor():
                                surv_preds = Survival.predict_survival_function(model, xte, times, n_post_samples=1000)
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
                            pycox_eval = EvalSurv(sanitized_surv_preds.T, sanitized_cvi['Survival_time'], sanitized_cvi['Event'], censor_surv="km")
                            c_index_cvi = pycox_eval.concordance_td()
                        except:    
                            c_index_cvi = np.nan
                        try:
                            lifelines_eval = LifelinesEvaluator(sanitized_surv_preds.T, sanitized_cvi['Survival_time'], sanitized_cvi['Event'],
                                                                ti_new[1]['Survival_time'], ti_new[1]['Event'])
                            median_survival_time = np.median(lifelines_eval.predict_time_from_curve(predict_median_survival_time))
                            mae_hinge_cvi = lifelines_eval.mae(method="Hinge")
                            d_calib = 1 if lifelines_eval.d_calibration()[0] > 0.05 else 0
                        except:
                            median_survival_time = np.nan
                            mae_hinge_cvi = np.nan
                            d_calib = np.nan
                        try:
                            brier_score_cvi = approx_brier_score(sanitized_cvi, sanitized_surv_preds)
                        except:
                            brier_score_cvi = np.nan
                            
                        if brier_score_cvi == np.inf:
                            brier_score_cvi = np.nan
                        
                        if mae_hinge_cvi > 1000:
                            mae_hinge_cvi = np.nan
                        
                        n_preds = len(sanitized_surv_preds)
                        event_detector_target = np.median(sanitized_cvi['Survival_time'])
                        t_total_split_time = time() - split_start_time

                        # Calculate the target datasheet TtE
                        if DATASET == 'xjtu':
                            datasheet_target = estimate_target_rul_xjtu(data_path, test, test_condition)
                        elif DATASET == 'pronostia':
                            datasheet_target = estimate_target_rul_pronostia(data_path, test, test_condition)

                        print(f"Evaluated {model_print_name} - {ft_selector_print_name} - {percentage}" +
                            f" - CI={round(c_index_cvi, 3)} - IBS={round(brier_score_cvi, 3)}" +
                            f" - MAE={round(mae_hinge_cvi, 3)} - DCalib={d_calib} - T={round(t_total_split_time, 3)}")

                        # Indexing the resul table
                        res_sr = pd.Series([model_print_name, ft_selector_print_name, c_index_cvi, brier_score_cvi,
                                            median_survival_time, mae_hinge_cvi, d_calib, event_detector_target, datasheet_target,
                                            n_preds, t_total_split_time, best_params, list(selected_fts), y_delta],
                                            index=["ModelName", "FtSelectorName", "CIndex", "BrierScore",
                                                   "MedianSurvTime", "MAEHinge", "DCalib", "EDTarget", "DatasheetTarget",
                                                   "Npreds", "TTotalSplit", "BestParams", "SelectedFts", "DeltaY"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                 
                # Indexing the file name linked to the DATASET condition
                if DATASET == "xjtu":
                    index = re.search(r"\d\d", cfg.RAW_DATA_PATH_XJTU[test_condition])
                    condition_name = cfg.RAW_DATA_PATH_XJTU[test_condition][index.start():-1] + "_" + str(int(CENSORING[censor_condition] * 100))
                elif DATASET == "pronostia":
                    index = re.search(r"\d\d", cfg.RAW_DATA_PATH_PRONOSTIA[test_condition])
                    condition_name = cfg.RAW_DATA_PATH_PRONOSTIA[test_condition][index.start():-1] + "_" + str(int(CENSORING[censor_condition] * 100))
                
                file_name = f"{model_name}_{condition_name}_results.csv"

                if TYPE == "correlated":
                    address = 'correlated'
                elif TYPE == "not_correlated":
                    address = 'not_correlated'
                else:
                    address = 'bootstrap'
                
                # Save the results to the proper DATASET type folder
                model_results.to_csv(f"data/logs/{DATASET}/{address}/" + file_name)

if __name__ == "__main__":
    main()
