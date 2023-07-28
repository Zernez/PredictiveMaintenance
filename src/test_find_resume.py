from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pathlib import Path
import config as cfg
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, VIF4, VIF8
from tools.regressors import CoxPH, CphRidge, CphLASSO, CphElastic, RSF, CoxBoost, GradientBoostingDART, WeibullAFT, LogNormalAFT, LogLogisticAFT, DeepSurv # XGBLinear, SVM
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.experiments import SurvivalRegressionCV
from sklearn.model_selection import train_test_split
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from time import time
import math
from auton_survival.metrics import survival_regression_metric
from auton_survival import DeepCoxPH
import argparse

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_REPEATS = 10
N_SPLITS = 3
N_ITER = 10
N_BOOT = 3
PLOT = True
RESUME = True
NEW_DATASET = False
#DATASET = "xjtu" # pronostia
#TYPE= "correlated" # not_correlated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    parser.add_argument('--corr', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()

    global DATASET
    global TYPE

    if args.dataset:
        DATASET = args.dataset

    if args.corr:
        TYPE = args.corr

    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap= N_BOOT)

    cov, boot, info_pack = FileReader(DATASET).read_data()
    
    X, y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
    #CoxPH, RSF, CoxBoost, DeepSurv, WeibullAFT
    models = [CoxPH, RSF, CoxBoost, DeepSurv, WeibullAFT]
    #NoneSelector, UMAP8, LowVar, SelectKBest4, SelectKBest8    
    ft_selectors = [NoneSelector, UMAP8, LowVar, SelectKBest4, SelectKBest8]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 0)
    T1, T2 = (X_train, y_train), (X_test, y_test)          

    print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors/{len(T1[0])} total samples. Dataset: {DATASET}. Type: {TYPE}")
    for model_builder in models:

        model_name = model_builder.__name__

        if model_name == 'WeibullAFT' or model_name == 'LogNormalAFT' or model_name == 'LogLogisticAFT' or model_name == 'ExponentialRegressionAFT':
            parametric = True
        else:
            parametric= False

        model_results = pd.DataFrame()
        for ft_selector_builder in ft_selectors:
            ft_selector_name = ft_selector_builder.__name__
            print ("ft_selector name: ", ft_selector_name )
            print ("model_builder name: ", model_name )

            for n_repeat in range(N_REPEATS):
                kf = KFold(n_splits=N_SPLITS, random_state= n_repeat, shuffle= True)
                for train, test in kf.split(T1[0], T1[1]):
                    split_start_time = time()

                    ti, cvi, ti_NN, cvi_NN = DataETL(DATASET).format_main_data_Kfold(T1, train, test)
                    ti, cvi, ti_NN , cvi_NN = DataETL(DATASET).centering_main_data(ti, cvi, ti_NN, cvi_NN)       

                    # Get current model and ft selector
                    if ft_selector_name == "NoneSelector":
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) None"
                    else:
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) {ft_selector_name}"
                    model_print_name = f"({models.index(model_builder)+1}) {model_name}"

                    #Create model instance and find best features
                    get_best_features_start_time = time()
                    model = model_builder().get_estimator()

                    if ft_selector_name == "UMAP8":
                            ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                            selected_fts = ft_selector.get_features()
                            ti_new = (selected_fts, ti[1])
                            ft_selector_NN = ft_selector_builder(ti_NN[0], ti_NN[1], estimator=model)
                            selected_fts = ft_selector_NN.get_features()
                            ti_new_NN = (selected_fts, ti_NN[1])
                            ft_selector = ft_selector_builder(cvi[0], cvi[1], estimator=model)
                            selected_fts = ft_selector.get_features()
                            cvi_new = (selected_fts, cvi[1])
                            ft_selector_NN = ft_selector_builder(cvi_NN[0], cvi_NN[1], estimator=model)
                            selected_fts = ft_selector_NN.get_features()
                            cvi_new_NN = (selected_fts, cvi_NN[1])
                            selected_fts= list(selected_fts.columns) 
                    elif ft_selector_name in ["RegMRMR4", "RegMRMR8"]:
                        if parametric== True:
                            y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                            ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model.lifelines_model)
                        else:
                            y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                            ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model)
                    
                    #From more 4 feature parametric models find difficulties to reach the convergence                                  
                    elif (parametric == True and ft_selector_name in ["NoneSelector", "SelectKBest8", "RegMRMR8", "LowVar", "VIF4"]):
                        c_index, brier_score = np.nan, np.nan
                        get_best_features_time, get_best_params_time, model_train_time = np.nan, np.nan, np.nan
                        model_ci_inference_time, model_bs_inference_time = np.nan, np.nan
                        t_total_split_time = np.nan
                        best_params, selected_fts = {}, []
                        res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, c_index, brier_score,
                                            get_best_features_time, get_best_params_time, model_train_time,
                                            model_ci_inference_time, model_bs_inference_time, t_total_split_time,
                                            best_params, selected_fts],
                                            index=["ModelName", "FtSelectorName", "NRepeat", "CIndex", "BrierScore",
                                                   "TBestFeatures", "TBestParams", "TModelTrain",
                                                   "TModelCIInference", "TModelBSInference", "TTotalSplit",
                                                   "BestParams", "SelectedFts"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                        continue
                    else:
                        ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)

                    if ft_selector_name == "UMAP8":     
                        get_best_features_time = time() - get_best_features_start_time 
                        print ("Created brand new features from UMAP")
                    else:
                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        ti_new[0].reset_index(inplace= True, drop=True)
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        cvi_new[0].reset_index(inplace= True, drop=True)
                        ti_new_NN =  (ti_NN[0].loc[:, selected_fts], ti_NN[1])
                        ti_new_NN[0].reset_index(inplace= True, drop=True)
                        cvi_new_NN = (cvi_NN[0].loc[:, selected_fts], cvi_NN[1])     
                        cvi_new_NN[0].reset_index(inplace= True, drop=True)                       
                        get_best_features_time = time() - get_best_features_start_time
                        print ("Selected features: ", selected_fts)

                    lower, upper = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [10, 90])
                    times = np.arange(math.ceil(lower), math.floor(upper))            

                    #Find hyperparams via CV
                    get_best_params_start_time = time()
                    space = model_builder().get_tuneable_params()
                    if parametric == True:
                        wf = model()
                        search = RandomizedSearchCV(wf, space, n_iter=N_ITER, cv=N_SPLITS, random_state=0)
                        x_ti_wf = pd.concat([ti_new[0].reset_index(drop=True),
                                            pd.DataFrame(ti_new[1]['Event'], columns=['Event'])], axis=1)
                        y_ti_wf = np.array([x[1] for x in ti_new[1]], float)
                        search.fit(x_ti_wf, y_ti_wf)
                        best_params = search.best_params_
                    elif model_name == "DeepSurv":
                        experiment = SurvivalRegressionCV(model='dcph', num_folds=N_SPLITS, hyperparam_grid= space)
                        model,best_params = experiment.fit(ti_new_NN[0], ti_new_NN[1], times, metric='ctd')              
                    else:
                        search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv= N_SPLITS, random_state= 0)
                        search.fit(ti_new[0], ti_new[1])
                        best_params = search.best_params_

                    get_best_params_time = time() - get_best_params_start_time

                    #Train on train set TI with new params
                    model_train_start_time = time()
                    if parametric == True:
                        model = search.best_estimator_
                        model.fit(x_ti_wf, y_ti_wf)
                    elif model_name == "DeepSurv":
                        model = DeepCoxPH(layers=[32, 32])
                        x= ti_new_NN[0].to_numpy()
                        t= ti_new_NN[1].loc[:,"time"].to_numpy()
                        e= ti_new_NN[1].loc[:,"event"].to_numpy()
                        model = model.fit(x, t, e, vsize=0.2, **best_params)
                    else:
                        model = search.best_estimator_
                        model.fit(ti_new[0], ti_new[1])
                    model_train_time = time() - model_train_start_time

                    #Get C-index scores from current CVI fold 
                    model_ci_inference_start_time = time()
                    if parametric == True:
                        x_cvi_wf = pd.concat([cvi_new[0].reset_index(drop=True),
                                              pd.DataFrame(cvi_new[1]['Event'],
                                                           columns=['Event'])], axis=1)
                        preds = model.predict(x_cvi_wf)
                        c_index = concordance_index(cvi[1]['Survival_time'], preds, cvi[1]['Event'])
                    elif model_name == "DeepSurv":
                        x= cvi_new_NN[0].to_numpy()
                        lower, upper = np.percentile(cvi_new[1][cvi_new[1].dtype.names[1]], [10, 90])
                        times = np.arange(math.ceil(lower), math.floor(upper)).tolist()
                        out_survival = model.predict_survival(x, max(times)).flatten()
                        if cvi_new_NN[1].isnull().values.any() or np.isnan(out_survival.any()):
                            c_index= np.nan
                            print ("Nan happened, skipped evalutation")
                        else:
                            c_index = survival_regression_metric('ctd', out_survival, cvi_new_NN[1], times=max(times))
                    else :
                        preds = model.predict(cvi_new[0])
                        c_index = concordance_index_censored(cvi[1]['Event'], cvi[1]['Survival_time'], preds)[0]
                    model_ci_inference_time = time() - model_ci_inference_start_time

                    #Get BS scores from current fold CVI fold
                    model_bs_inference_start_time = time()
                    if parametric == True:
                        model_instance = model.lifelines_model
                        surv_prob = model_instance.predict_survival_function(cvi_new[0]).T
                        brier_score = approx_brier_score(cvi_new[1], surv_prob)
                    elif model_name == "DeepSurv":
                        lower, upper = np.percentile(cvi_new[1][cvi_new[1].dtype.names[1]], [10, 90])
                        times = np.arange(math.ceil(lower), math.floor(upper)).tolist()
                        x = cvi_new_NN[0].to_numpy()
                        out_survival = model.predict_survival(x, times)


                        if cvi_new_NN[1].isnull().values.any():
                            brier_score= np.nan
                            print ("Nan happened, skipped evalutation")
                        else:
                            brier_score = np.mean(survival_regression_metric('brs', out_survival, cvi_new_NN[1], times=times))
                            
                    elif model_name == "SVM":
                        brier_score = np.nan
                    else:
                        surv_probs = pd.DataFrame(np.row_stack([fn(times)
                                                                for fn in model.predict_survival_function(cvi_new[0])]))
                        brier_score = approx_brier_score(cvi_new[1], surv_probs)
                    model_bs_inference_time = time() - model_bs_inference_start_time

                    t_total_split_time = time() - split_start_time
                    print(f"Evaluated {model_print_name} - {ft_selector_print_name}" + \
                          f" - CI={round(c_index, 3)} - BS={round(brier_score, 3)} - T={round(t_total_split_time, 3)}")

                    res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, c_index, brier_score,
                                        get_best_features_time, get_best_params_time, model_train_time,
                                        model_ci_inference_time, model_bs_inference_time, t_total_split_time,
                                        best_params, selected_fts],
                                        index=["ModelName", "FtSelectorName", "NRepeat", "CIndex", "BrierScore",
                                               "TBestFeatures", "TBestParams", "TModelTrain",
                                               "TModelCIInference", "TModelBSInference", "TTotalSplit",
                                               "BestParams", "SelectedFts"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)

        file_name = f"{model_name}_results.csv"
        model_results.to_csv(f"data/logs/{DATASET}/{TYPE}/" + file_name)

if __name__ == "__main__":
    main()
