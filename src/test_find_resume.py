import numpy as np
import pandas as pd
from time import time
import math
import argparse
import warnings
import config as cfg
import re
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, VIF4, VIF8, PHSelector
from tools.regressors import CoxPH, CphRidge, CphLASSO, CphElastic, RSF, CoxBoost, GradientBoostingDART, WeibullAFT, LogNormalAFT, LogLogisticAFT, DeepSurv, DSM
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.experiments import SurvivalRegressionCV
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from lifelines import WeibullAFTFitter

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = 3
PLOT = True
RESUME = True
NEW_DATASET = False
N_REPEATS = 1
N_INTERNAL_SPLITS = 3 
N_ITER = 1

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

    if args.dataset:
        DATASET = args.dataset

    if args.typedata:
        TYPE = args.typedata
    
    if args.merge:
        MERGE = args.merge

    if DATASET == "xjtu":
        N_CONDITION = len (cfg.RAW_DATA_PATH_XJTU)
        N_BEARING = cfg.N_REAL_BEARING_XJTU
        N_SPLITS = 3  
    elif DATASET == "pronostia":
        N_CONDITION = len (cfg.RAW_DATA_PATH_PRONOSTIA)
        N_BEARING = cfg.N_REAL_BEARING_PRONOSTIA
        N_SPLITS = 2
    
    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap=N_BOOT)
    
    models = [CoxPH, RSF, CoxBoost, DeepSurv, DSM, WeibullAFT]
    ft_selectors = [NoneSelector, PHSelector]
    survival = Survival()    
    
    cov_group = []
    boot_group = []
    info_group = []
    for i in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET).read_data(i)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    data_container_X = []
    data_container_y= []
    if MERGE == True:
        data_X = pd.DataFrame()
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, data_temp_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            if i== 0:
                data_y_merge =  data_temp_y
            else:
                data_y_merge =  np.concatenate((data_y_merge, data_temp_y))
            data_X_merge = pd.concat([data_X_merge, data_temp_X], ignore_index=True)
        data_container_X.append(data_X_merge)
        data_container_y.append(data_y_merge)
    else:
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, data_temp_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            data_container_X.append(data_temp_X)
            data_container_y.append(data_temp_y)
                                                                          
    for i, data_X, data_y in enumerate(zip(data_container_X, data_container_y)):

        y_delta = np.delete(data_y, slice(None))
        total_upsampling_fold= cfg.N_BOOT_FOLD_UPSAMPLING 

        for element in range(0, N_BEARING * total_upsampling_fold, total_upsampling_fold):
                y_delta = np.append(y_delta, data_y[element])

        T1, T2 = (data_X, data_y), (data_X, data_y)

        print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors/{len(T1[0])} total samples. Dataset: {DATASET}. Type: {TYPE}")
        for model_builder in models:

            model_name = model_builder.__name__

            if model_name == 'WeibullAFT' or model_name == 'LogNormalAFT' or model_name == 'LogLogisticAFT' or model_name == 'ExponentialRegressionAFT':
                parametric = True
            else:
                parametric = False

            model_results = pd.DataFrame()
            for ft_selector_builder in ft_selectors:
                ft_selector_name = ft_selector_builder.__name__
                print("ft_selector name: ", ft_selector_name)
                print("model_builder name: ", model_name)

                dummy_x= list(range(0, N_BEARING))
                dummy_y= list(range(0, N_BEARING))

                for n_repeat in range(N_REPEATS):
                    kf = KFold(n_splits= N_SPLITS, random_state=n_repeat, shuffle=True)
                    for train, test in kf.split(dummy_x, dummy_y):     

                        train_index = train
                        train = np.delete(train, slice(None))
                        test_index = test
                        test = np.delete(test, slice(None))                                       

                        for element in train_index:
                            for boot_index in range(element * total_upsampling_fold, (element * total_upsampling_fold) + total_upsampling_fold, 1):
                                train = np.append(train, boot_index)

                        for element in test_index:
                            for boot_index in range(element * total_upsampling_fold, (element * total_upsampling_fold) + total_upsampling_fold, 1):
                                test = np.append(test, boot_index)

                        split_start_time = time()

                        ti, cvi, ti_NN, cvi_NN = DataETL(
                            DATASET).format_main_data_Kfold(T1, train, test)
                        ti, cvi, ti_NN, cvi_NN = DataETL(
                            DATASET).centering_main_data(ti, cvi, ti_NN, cvi_NN)

                        ft_selector_print_name = f"{ft_selector_name}"
                        model_print_name = f"{model_name}"
                        
                        # Create model instance and find best features
                        get_best_features_start_time = time()
                        model = model_builder().get_estimator()
                        if ft_selector_name == "PHSelector":
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=[DATASET, TYPE])
                        else:
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)

                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        ti_new[0].reset_index(inplace=True, drop=True)
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        cvi_new[0].reset_index(inplace=True, drop=True)
                        ti_new_NN =  (ti_NN[0].loc[:, selected_fts], ti_NN[1])
                        ti_new_NN[0].reset_index(inplace=True, drop=True)
                        cvi_new_NN = (cvi_NN[0].loc[:, selected_fts], cvi_NN[1])     
                        cvi_new_NN[0].reset_index(inplace=True, drop=True)
                        get_best_features_time = time() - get_best_features_start_time

                        # Set event times
                        lower, upper = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [10, 90])
                        times = np.arange(math.ceil(lower), math.floor(upper)).tolist()

                        # Find hyperparams via CV
                        get_best_params_start_time = time()
                        space = model_builder().get_tuneable_params()
                        if parametric == True:
                            wf = model()
                            search = RandomizedSearchCV(
                                wf, space, n_iter=N_ITER, cv=N_INTERNAL_SPLITS, random_state=0)
                            x_ti_wf = pd.concat([ti_new[0].reset_index(drop=True),
                                                pd.DataFrame(ti_new[1]['Event'], columns=['Event'])], axis=1)
                            y_ti_wf = np.array([x[1] for x in ti_new[1]], float)
                            search.fit(x_ti_wf, y_ti_wf)
                            best_params = search.best_params_
                        elif model_name == "DeepSurv":
                            experiment = SurvivalRegressionCV(model='dcph', num_folds=N_INTERNAL_SPLITS, hyperparam_grid=space)
                            model, best_params = experiment.fit(ti_new_NN[0], ti_new_NN[1], times, metric='brs')
                        elif model_name == "DSM":
                            experiment = SurvivalRegressionCV(model='dsm', num_folds=N_INTERNAL_SPLITS, hyperparam_grid=space)
                            model, best_params = experiment.fit(ti_new_NN[0], ti_new_NN[1], times, metric='brs')                
                        else:
                            search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv=N_INTERNAL_SPLITS, random_state=0)
                            search.fit(ti_new[0], ti_new[1])
                            best_params = search.best_params_

                        get_best_params_time = time() - get_best_params_start_time

                        # Train on train set TI with new params
                        model_train_start_time = time()
                        if parametric == True:
                            x_ti_wf = pd.concat([ti_new[0].reset_index(drop=True),
                                                pd.DataFrame(ti_new[1]['Survival_time'],
                                                            columns=['Survival_time'])], axis=1)
                            x_ti_wf = pd.concat([x_ti_wf.reset_index(drop=True),
                                                pd.DataFrame(ti_new[1]['Event'],
                                                            columns=['Event'])], axis=1)
                            model= WeibullAFTFitter(**best_params)
                            model.fit(x_ti_wf, duration_col='Survival_time', event_col='Event')
                        elif model_name == "DeepSurv":
                            model = DeepCoxPH(layers=[32, 32])
                            x = ti_new_NN[0].to_numpy()
                            t = ti_new_NN[1].loc[:, "time"].to_numpy()
                            e = ti_new_NN[1].loc[:, "event"].to_numpy()
                            model = model.fit(x, t, e, vsize=0.3, **best_params)
                        elif model_name == "DSM":
                            model = DeepSurvivalMachines(layers=[32, 32])
                            x = ti_new_NN[0].to_numpy()
                            t = ti_new_NN[1].loc[:, "time"].to_numpy()
                            e = ti_new_NN[1].loc[:, "event"].to_numpy()
                            model = model.fit(x, t, e, vsize=0.3, **best_params)
                        else:
                            model = search.best_estimator_
                            model.fit(ti_new[0], ti_new[1])
                        model_train_time = time() - model_train_start_time

                        # Get C-index scores from current CVI fold 
                        model_ci_inference_start_time = time()
                        if parametric == True:
                            x_cvi_wf = pd.concat([cvi_new[0].reset_index(drop=True),
                                                pd.DataFrame(cvi_new[1]['Survival_time'],
                                                            columns=['Survival_time'])], axis=1)
                            x_cvi_wf = pd.concat([x_cvi_wf.reset_index(drop=True),
                                                pd.DataFrame(cvi_new[1]['Event'],
                                                            columns=['Event'])], axis=1)
                            preds = survival.predict_survival_function(
                                model, x_cvi_wf, times)
                            ev = EvalSurv(
                                preds.T, cvi[1]['Survival_time'], cvi[1]['Event'], censor_surv="km")
                            c_index = ev.concordance_td()
                        elif model_name == "DeepSurv" or model_name == "DSM":
                            xte = cvi_new_NN[0].to_numpy()
                            preds = survival.predict_survival_function(model, xte, times)
                            ev = EvalSurv(preds.T, cvi_new[1]['Survival_time'], cvi_new[1]['Event'],
                                        censor_surv="km")
                            c_index = ev.concordance_td()
                        else:
                            preds = survival.predict_survival_function(model, cvi_new[0], times)
                            ev = EvalSurv(preds.T, cvi_new[1]['Survival_time'], cvi_new[1]['Event'],
                                        censor_surv="km")
                            c_index = ev.concordance_td()
                        model_ci_inference_time = time() - model_ci_inference_start_time

                        # Get BS scores from current fold CVI fold
                        model_bs_inference_start_time = time()
                        if parametric == True:
                            brier_score = approx_brier_score(cvi_new[1], preds)
                            nbll = np.mean(ev.nbll(np.array(times)))
                        elif model_name == "DeepSurv": 
                            NN_surv_probs = model.predict_survival(xte)
                            brier_score = approx_brier_score(cvi_new[1], NN_surv_probs)
                            nbll = np.mean(ev.nbll(np.array(times)))                   
                        elif model_name == "DSM":
                            NN_surv_probs = pd.DataFrame(model.predict_survival(xte, t=times))
                            brier_score = approx_brier_score(cvi_new[1], NN_surv_probs)
                            nbll = np.mean(ev.nbll(np.array(times)))
                        else:
                            surv_probs = pd.DataFrame(preds)
                            brier_score = approx_brier_score(cvi_new[1], surv_probs)
                            nbll = np.mean(ev.nbll(np.array(times)))

                        model_bs_inference_time = time() - model_bs_inference_start_time
                        t_total_split_time = time() - split_start_time

                        print(f"Evaluated {model_print_name} - {ft_selector_print_name}" +
                            f" - CI={round(c_index, 3)} - BS={round(brier_score, 3)} - NBLL={round(nbll, 3)} - T={round(t_total_split_time, 3)}")

                        res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, c_index, brier_score, nbll,
                                            get_best_features_time, get_best_params_time, model_train_time,
                                            model_ci_inference_time, model_bs_inference_time, t_total_split_time,
                                            best_params, selected_fts, y_delta],
                                        index=["ModelName", "FtSelectorName", "NRepeat", "CIndex", "BrierScore", "NBLL",
                                                "TBestFeatures", "TBestParams", "TModelTrain",
                                                "TModelCIInference", "TModelBSInference", "TTotalSplit",
                                                "BestParams", "SelectedFts", "DeltaY"])
                        model_results = pd.concat(
                            [model_results, res_sr.to_frame().T], ignore_index=True)
            
            index = re.search(r"\d\d", cfg.RAW_DATA_PATH_PRONOSTIA[i])
            type_name = cfg.RAW_DATA_PATH_PRONOSTIA[i][index.start():-1]
            
            file_name = f"{model_name}_{type_name}_results.csv"

            if TYPE == "correlated":
                address = 'correlated'
            elif TYPE == "not_correlated":
                address = 'not_correlated'
            else:
                address = 'bootstrap'

            model_results.to_csv(f"data/logs/{DATASET}/{address}/" + file_name)


if __name__ == "__main__":
    main()
