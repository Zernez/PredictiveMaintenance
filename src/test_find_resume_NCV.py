import numpy as np
import pandas as pd
from time import time
import math
import argparse
import warnings
import config as cfg
import re
import os
from pycox.evaluation import EvalSurv
from scipy.integrate import trapezoid
from sksurv.util import Surv
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, VIF4, VIF8, PHSelector
from tools.regressors import CoxPH, CphRidge, CphLASSO, CphElastic, RSF, CoxBoost, GradientBoostingDART, WeibullAFT, LogNormalAFT, LogLogisticAFT, DeepSurv, DSM, BNNmcd
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.experiments import SurvivalRegressionCV
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from lifelines import WeibullAFTFitter
import logging
import contextlib

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = cfg.N_BOOT
PLOT = True
RESUME = True
NEW_DATASET = False
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
    parser.add_argument('--merge', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()

    global DATASET
    global TYPE
    global MERGE
    global N_CONDITION
    global N_BEARING
    global N_SPLITS
    global TRAIN_SIZE
    global CENSORING

    if args.dataset:
        DATASET = args.dataset
        cfg.DATASET_NAME = args.dataset

    if args.typedata:
        TYPE = args.typedata
    
    if args.merge:
        MERGE = args.merge

    #DATASET= "xjtu"
    #TYPE= "correlated"
    #MERGE= "False"

    if TYPE == "bootstrap":
        cfg.N_BOOT = 8
        cfg.DATA_TYPE = "bootstrap"
    else:
        cfg.DATA_TYPE = "not_bootstrap"        

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
    
    #For the first time running, a NEW_DATASET is needed
    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap=N_BOOT)   
    #Insert the models and feature name selector for CV hyperparameter search
    models = [CoxPH, RSF, DeepSurv, DSM, BNNmcd]
    ft_selectors = [NoneSelector]
    survival = Survival()
    data_util = DataETL(DATASET)

    #Extract information from the dataset selected from the config file
    cov_group = []
    boot_group = []
    info_group = []
    for i in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET).read_data(i)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    #Transform information from the dataset selected from the config file
    data_container_X = []
    data_container_y= []
    if MERGE == True:
        data_X_merge = pd.DataFrame()
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_temp_y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            if i== 0:
                deltaref_y_merge =  deltaref_temp_y
            else:
                deltaref_y_merge =  deltaref_y_merge.update(deltaref_temp_y)
            data_X_merge = pd.concat([data_X_merge, data_temp_X], ignore_index=True)
        data_container_X.append(data_X_merge)
        data_container_y.append(deltaref_y_merge)
    else:
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            data_container_X.append(data_temp_X)
            data_container_y.append(deltaref_y)

    #Load information from the dataset selected in the config file                                                                          
    for i, (data, data_y) in enumerate(zip(data_container_X, data_container_y)):

        #Information about the event estimation in event detector
        y_delta = data_y

        #Iteration for each censored condition
        for j, percentage in enumerate(CENSORING):

            #Eventually control the censored data by CENSORING
            data_X= []
            for data_group in data:   
                data_temp_X  = data_util.control_censored_data(data_group, percentage= percentage)
                data_X.append(data_temp_X)

            #Indexing the dataset to avoid train/test leaking
            dummy_x = list(range(0, int(np.floor(N_BEARING * TRAIN_SIZE)), 1))                       
            
            print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors. Dataset: {DATASET}. Type: {TYPE}")
            
            #For all models selected
            for model_builder in models:
                model_name = model_builder.__name__
                model_results = pd.DataFrame()

                #For all feature selector selected
                for ft_selector_builder in ft_selectors:
                    ft_selector_name = ft_selector_builder.__name__
                    
                    #For N_SPLITS folds 
                    kf = KFold(n_splits= N_SPLITS, shuffle= False)
                    for split_idx, (train, test) in enumerate(kf.split(dummy_x)):
                        #Start take the time for search the hyperparameters for each fold
                        split_start_time = time()

                        #Load the train data from group indexed Kfold splitting avoiding train/test leaking                
                        data_X_merge = pd.DataFrame()
                        for element in train:
                            data_X_merge = pd.concat([data_X_merge, data_X [element]], ignore_index=True)
                        data_y = Surv.from_dataframe("Event", "Survival_time", data_X_merge)
                        T1 = (data_X_merge, data_y)

                        #Load the test data from group indexed Kfold splitting avoiding train/test leaking               
                        data_X_merge = pd.DataFrame()
                        for element in test:
                            data_X_merge = pd.concat([data_X_merge, data_X [element]], ignore_index=True)
                        data_y = Surv.from_dataframe("Event", "Survival_time", data_X_merge)
                        T2 = (data_X_merge, data_y)

                        #Fromat and center the data     
                        ti, cvi, ti_NN, cvi_NN = DataETL(DATASET).format_main_data (T1, T2)
                        ti, cvi, ti_NN, cvi_NN = DataETL(DATASET).centering_main_data (ti, cvi, ti_NN, cvi_NN)

                        ft_selector_print_name = f"{ft_selector_name}"
                        model_print_name = f"{model_name}"
                        
                        #Create model instance and find best features
                        model = model_builder().get_estimator()
                        if ft_selector_name == "PHSelector":
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=[DATASET, TYPE])
                        else:
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                        
                        #Format the data with the feature selected
                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        ti_new[0].reset_index(inplace=True, drop=True)
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        cvi_new[0].reset_index(inplace=True, drop=True)
                        ti_new_NN =  (ti_NN[0].loc[:, selected_fts], ti_NN[1])
                        ti_new_NN[0].reset_index(inplace=True, drop=True)
                        cvi_new_NN = (cvi_NN[0].loc[:, selected_fts], cvi_NN[1])     
                        cvi_new_NN[0].reset_index(inplace=True, drop=True)

                        #Set event times
                        lower, upper = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [0, 100])
                        times = np.arange(math.ceil(lower), math.floor(upper)).tolist()

                        #Find hyperparams via CV from hyperparamters' space
                        space = model_builder().get_tuneable_params()
                        if model_name == "DeepSurv":
                            experiment = SurvivalRegressionCV(model='dcph', num_folds=N_INTERNAL_SPLITS, hyperparam_grid=space)
                            model, best_params = experiment.fit(ti_new_NN[0], ti_new_NN[1], times, metric='ctd')
                        elif model_name == "DSM":
                            experiment = SurvivalRegressionCV(model='dsm', num_folds=N_INTERNAL_SPLITS, hyperparam_grid=space)
                            model, best_params = experiment.fit(ti_new_NN[0], ti_new_NN[1], times, metric='ctd')
                        elif model_name == "BNNmcd":
                            param_list = list(ParameterSampler(space, n_iter=N_ITER, random_state=0))
                            sample_results = pd.DataFrame()
                            for sample in param_list:
                                param_results = pd.DataFrame()
                                kf = KFold(n_splits=N_INTERNAL_SPLITS, random_state=0, shuffle=True)
                                for split_idx, (train_in, test_in) in enumerate(kf.split(ti_new_NN[0], ti_new_NN[1])):
                                    t_train = np.array(ti_new_NN[1].iloc[train_in]["time"])
                                    e_train = np.array(ti_new_NN[1].iloc[train_in]["event"])
                                    t_test = np.array(ti_new_NN[1].iloc[test_in]["time"])
                                    e_test = np.array(ti_new_NN[1].iloc[test_in]["event"])
                                    f_train =  np.array(ti_new_NN[0].iloc[train_in])
                                    f_test =  np.array(ti_new_NN[0].iloc[test_in])
                                    model = model_builder().make_model(sample)
                                    model.fit(f_train, t_train, e_train)
                                    lower, upper = np.percentile(t_train, [0, 100])
                                    times = np.arange(math.ceil(lower), math.floor(upper)).tolist()
                                    preds = model.predict_survival(f_test, times)
                                    preds = pd.DataFrame(np.mean(preds, axis=0))
                                    ev = EvalSurv(preds.T, t_test, e_test, censor_surv="km")
                                    c_index_ti = ev.concordance_td()
                                    res_sr = pd.Series([str(model_name), split_idx, sample, c_index_ti],
                                                    index=["ModelName", "SplitIdx", "Params", "CIndex"])
                                    param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
                                mean_c_index_ti = param_results['CIndex'].mean()
                                res_sr = pd.Series([str(model_name), param_results.iloc[0]["Params"], mean_c_index_ti],
                                                index=["ModelName", "Params", "CIndex"])
                                sample_results = pd.concat([sample_results, res_sr.to_frame().T], ignore_index=True)
                            best_params = sample_results.loc[sample_results['CIndex'].astype(float).idxmax()]['Params']
                        else:
                            search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv=N_INTERNAL_SPLITS, random_state=0)
                            search.fit(ti_new[0], ti_new[1])
                            best_params = search.best_params_

                        #Train on train set TI with new parameters
                        x = ti_new_NN[0].to_numpy()
                        t = ti_new_NN[1].loc[:, "time"].to_numpy()
                        e = ti_new_NN[1].loc[:, "event"].to_numpy()
                        if model_name == "DeepSurv":
                            model = DeepCoxPH(layers=[32])
                            with open(os.devnull, 'w') as devnull:
                                with contextlib.redirect_stdout(devnull):
                                    model = model.fit(x, t, e, vsize=0.3, **best_params)
                        elif model_name == "DSM":
                            model = DeepSurvivalMachines(layers=[32])
                            with open(os.devnull, 'w') as devnull:
                                with contextlib.redirect_stdout(devnull):
                                    model = model.fit(x, t, e, vsize=0.3, **best_params)
                        elif model_name == "BNNmcd":
                            model = model_builder().make_model(best_params)
                            model.fit(x, t, e)
                        else:
                            model = search.best_estimator_
                            model.fit(ti_new[0], ti_new[1])
                        
                        #Set the time range for calculate the survivor function 
                        lower, upper = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [0, 100])
                        times_cvi = np.arange(0, math.floor(upper)).tolist()
                        
                        #Set the time range for calculate the survivor function for NN
                        lower_NN, upper_NN = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [0, 100])
                        times_cvi_NN = np.arange(0, math.floor(upper_NN)).tolist()

                        #Get C-index scores from current CVI fold 
                        if model_name == "DeepSurv" or model_name == "DSM":
                            xte = cvi_new_NN[0].to_numpy()
                            surv_preds = survival.predict_survival_function(model, xte, times_cvi_NN)
                            surv_preds.replace(np.nan, 1e-1000, inplace=True)
                            surv_preds[math.ceil(upper)] = 1e-1000
                            surv_preds.reset_index(drop=True, inplace=True)
                            ev = EvalSurv(surv_preds.T, cvi_new[1]['Survival_time'], cvi_new[1]['Event'], censor_surv="km")
                            c_index_cvi = ev.concordance_td()
                        elif model_name == "BNNmcd":
                            xte = cvi_new_NN[0].to_numpy()       
                            surv_preds = survival.predict_survival_function(model, xte, times_cvi_NN)
                            surv_preds.replace(np.nan, 1e-1000, inplace=True)
                            surv_preds[math.ceil(upper)] = 1e-1000
                            surv_preds.reset_index(drop=True, inplace=True)
                            ev = EvalSurv(surv_preds.T, cvi_new_NN[1]['time'].to_numpy(), cvi_new_NN[1]['event'].to_numpy(), censor_surv="km")
                            c_index_cvi = ev.concordance_td()
                        else:
                            surv_preds = survival.predict_survival_function(model, cvi_new[0], times_cvi)
                            surv_preds.replace(np.nan, 1e-1000, inplace=True)
                            surv_preds[math.ceil(upper)] = 1e-1000
                            surv_preds.reset_index(drop=True, inplace=True)
                            ev = EvalSurv(surv_preds.T, cvi_new[1]['Survival_time'], cvi_new[1]['Event'], censor_surv="km")
                            c_index_cvi = ev.concordance_td()

                        #Get BS and NBLL scores from current fold CVI fold and expectation of TtE integration by the median of all test set
                        if model_name == "DeepSurv":
                            NN_surv_probs = model.predict_survival(xte)
                            brier_score_cvi = approx_brier_score(cvi_new[1], NN_surv_probs)
                            nbll_cvi = np.mean(ev.nbll(np.array(times_cvi_NN)))
                            sd_preds = np.std(surv_preds)
                            n_preds = len(surv_preds)
                            med_surv_preds = surv_preds.median()
                            event_detector_target = np.median(cvi_new[1]['Survival_time'])
                            surv_expect= trapezoid(y= med_surv_preds.values, x= med_surv_preds.index)
                        elif model_name == "DSM":
                            NN_surv_probs = pd.DataFrame(model.predict_survival(xte, t= times_cvi_NN))
                            brier_score_cvi = approx_brier_score(cvi_new[1], NN_surv_probs)
                            nbll_cvi = np.mean(ev.nbll(np.array(times_cvi_NN)))
                            sd_preds = np.std(surv_preds)
                            n_preds = len(surv_preds)
                            med_surv_preds = surv_preds.median()
                            event_detector_target = np.median(cvi_new[1]['Survival_time'])
                            surv_expect= trapezoid(y= med_surv_preds.values, x= med_surv_preds.index)
                        elif model_name == "BNNmcd":
                            brier_score_cvi = approx_brier_score(cvi_new[1], surv_probs)
                            nbll_cvi = np.mean(ev.nbll(np.array(times)))
                            sd_preds = np.std(surv_preds)
                            n_preds = len(surv_preds)
                            med_surv_preds = surv_preds.median()
                            event_detector_target = np.median(cvi_new[1]['Survival_time'])
                            surv_expect= trapezoid(y= med_surv_preds.values, x= med_surv_preds.index)                                
                        else:
                            surv_probs = pd.DataFrame(surv_preds)
                            brier_score_cvi = approx_brier_score(cvi_new[1], surv_probs)
                            nbll_cvi = np.mean(ev.nbll(np.array(times)))
                            sd_preds = np.std(surv_preds)
                            n_preds = len(surv_preds)
                            med_surv_preds = surv_preds.median()
                            event_detector_target = np.median(cvi_new[1]['Survival_time'])
                            surv_expect= trapezoid(y= med_surv_preds.values, x= med_surv_preds.index)

                        t_total_split_time = time() - split_start_time

                        #Prepare settings for calculate the target datasheet TtE
                        if DATASET == 'xjtu':
                            dataset_tte = DATASET.upper() + '-SY'
                            type_test = data_path[i]
                            index = re.search(r"\d\d", type_test)
                            info_type_test = type_test[index.start():-1]
                        else:
                            dataset_tte = DATASET.upper()
                            type_test = data_path[i]
                            index = re.search(r"\d\d", type_test)
                            info_type_test = type_test[index.start():-1]
                        
                        #Calculate the target TtE for the test data from the datasheet
                        temp_tte = []
                        itr_tte = os.walk(f"./data/{dataset_tte}/{info_type_test}")
                        next(itr_tte)
                        iter_tte = 0
                        for next_root, next_dirs, next_files in itr_tte:
                            if iter_tte in test:
                                temp_tte.append(len([f for f in os.listdir(next_root) if os.path.isfile(os.path.join(next_root, f))]))
                            iter_tte += 1
                            
                        #Makes a unique value of TtE for the test data
                        datasheet_target = np.median(temp_tte)

                        print(f"Evaluated {model_print_name} - {ft_selector_print_name} - {percentage}" +
                            f" - CI={round(c_index_cvi, 3)} - BS={round(brier_score_cvi, 3)}" +
                            f" - NBLL={round(nbll_cvi, 3)} - T={round(t_total_split_time, 3)}")

                        #Indexing the resul table
                        res_sr = pd.Series([model_print_name, ft_selector_print_name, c_index_cvi, brier_score_cvi, nbll_cvi, 
                                            surv_expect, event_detector_target, datasheet_target, sd_preds, n_preds, t_total_split_time,
                                            best_params, selected_fts, y_delta],
                                            index=["ModelName", "FtSelectorName", "CIndex", "BrierScore", "NBLL",
                                                    "SurvExpect", "EDTarget", "DatasheetTarget", "SDTtE", "Npreds", "TTotalSplit",
                                                    "BestParams", "SelectedFts", "DeltaY"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                
                #Indexing the file name linked to the DATASET condition
                if DATASET == "xjtu":
                    index = re.search(r"\d\d", cfg.RAW_DATA_PATH_XJTU[i])
                    condition_name = cfg.RAW_DATA_PATH_XJTU[i][index.start():-1] + "_" + str(int(CENSORING[j] * 100))
                elif DATASET == "pronostia":
                    index = re.search(r"\d\d", cfg.RAW_DATA_PATH_PRONOSTIA[i])
                    condition_name = cfg.RAW_DATA_PATH_XJTU[i][index.start():-1] + "_" + str(int(CENSORING[j] * 100))                
                
                file_name = f"{model_name}_{condition_name}_results.csv"

                if TYPE == "correlated":
                    address = 'correlated'
                elif TYPE == "not_correlated":
                    address = 'not_correlated'
                else:
                    address = 'bootstrap'
                
                #Save the results to the proper DATASET type folder
                model_results.to_csv(f"data/logs/{DATASET}/{address}/" + file_name)

if __name__ == "__main__":
    main()
