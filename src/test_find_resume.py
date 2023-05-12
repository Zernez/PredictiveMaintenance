from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tools import file_writer
from pathlib import Path
import config as cfg
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, RFE4 ,RFE8 , SFS4, SFS8
from tools.regressors import Cph, CphRidge, CphLasso, CphElastic, RSF, CoxBoost, WeibullAFT, LogNormalAFT, LogLogisticAFT, ExponentialAFT, XGBTree, XGBDart, SVM, GradientBoosting, GradientBoostingDART # XGBLinear
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from sklearn.model_selection import train_test_split
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from time import time
import math
import pickle

N_REPEATS = 3
N_SPLITS = 3
N_ITER = 3
N_BOOT = 2
PLOT = True
RESUME = True

def main():

    cov, boot, info_pack = FileReader().read_data_xjtu()
    
#    df_surv = data_ETL.DataETL().make_covariates(df)
    X, y = DataETL().make_surv_data_sklS(cov, boot, info_pack, N_BOOT)

    models = [GradientBoostingDART] #  XGBLinear, ExponentialAFT      WeibullAFT, LogNormalAFT, LogLogisticAFT, Cph, CphRidge, CphLasso, CphElastic,  RSF, CoxBoost, XGBTree, XGBDart, SVM
    ft_selectors = [NoneSelector, UMAP8, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, SFS4, SFS8, RFE4, RFE8] #NoneSelector, UMAP6, LowVar, SelectKBest4, SelectKBest8, SFS4, SFS8, RegMRMR4, RegMRMR8
    #ft_selectors = [NoneSelector, LowVar,RFE4]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    T1, T2 = (X_train, y_train), (X_test, y_test)

    # X_trainSD = X_train[['Event', 'Survival_time']].copy()
    # X_train.drop(["Event", "Survival_time"], axis=1)
    # X_testSD = X_train[['Event', 'Survival_time']].copy()
    # X_test.drop(["Event", "Survival_time"], axis=1)     
    # y_trainSD = X_train[['Event', 'Survival_time']].copy()
    # y_train.drop(["Event", "Survival_time"], axis=1)
    # y_testSD = X_train[['Event', 'Survival_time']].copy()
    # y_test.drop(["Event", "Survival_time"], axis=1)             

    print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors/{len(T1[0])} total samples")
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
                kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=n_repeat)
                for train, test in kf.split(T1[0], T1[1]):
                    split_start_time = time()

                    # # Split data
                    # ti = (T1[0].iloc[train], T1[1][train])
                    # cvi = (T1[0].iloc[test], T1[1][test])

                    ti, cvi, ti_NN, y_ti_NN, cvi_NN, y_cvi_NN, val_NN = DataETL().centering_and_NNcloning(T1, T2, train, test)

                    # Get current model and ft selector
                    if ft_selector_name == "NoneSelector":
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) None"
                    else:
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) {ft_selector_name}"
                    model_print_name = f"({models.index(model_builder)+1}) {model_name}"

                    # Create model instance and find best features
                    get_best_features_start_time = time()
                    model = model_builder().get_estimator()
                    model_class_name = model.__class__.__name__

                    if parametric == False and ft_selector_name in ["RegMRMR4", "RegMRMR8"]:
                        y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                        ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model)
                    elif ft_selector_name == "UMAP8":
                        ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                        selected_fts = ft_selector.get_features()
                        ti_new = (selected_fts, ti[1])
                        ft_selector = ft_selector_builder(cvi[0], cvi[1], estimator=model)
                        selected_fts = ft_selector.get_features()
                        cvi_new = (selected_fts, cvi[1])
                        selected_fts= list(selected_fts.columns)                     
                    elif parametric == True and ft_selector_name in ["NoneSelector", "RFE4", "RFE8", "SFS4", "SFS8", "SelectKBest8", "RegMRMR8"]:
                        # No support for parametric and some selectors, so skip runs
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
                    elif parametric == True and ft_selector_name in ["RegMRMR4"]:
                        y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                        ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model.lifelines_model)
                    elif model_class_name == "XGBRegressor" and ft_selector_name in ["RFE4", "RFE8"]:
                        y_ti_xgb = [x[1] if x[0] else -x[1] for x in ti[1]]
                        ft_selector = ft_selector_builder(ti[0], y_ti_xgb, estimator=model)
                    elif model_class_name == "DeepSurv":
                        ft_selector = ft_selector_builder(ti_NN, y_ti_NN, estimator=model)
                    else:
                        ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)

                    if ft_selector_name != "UMAP8":     
                        selected_fts = ft_selector.get_features()
                        ti_new =  (ti[0].loc[:, selected_fts], ti[1])
                        cvi_new = (cvi[0].loc[:, selected_fts], cvi[1])
                        ti_new_NN =  (ti_NN.loc[:, selected_fts], y_ti_NN)
                        cvi_new_NN = (cvi_NN.loc[:, selected_fts], y_cvi_NN)
                        get_best_features_time = time() - get_best_features_start_time
                        print ("Selected features: ", selected_fts)
                    else:
                        get_best_features_time = time() - get_best_features_start_time 
                        print ("Created brand new features from UMAP")

                    lower, upper = np.percentile(ti_new[1][ti_new[1].dtype.names[1]], [10, 90])
                    times = np.arange(math.ceil(lower), math.floor(upper+1))

                    # Find hyperparams via CV
                    get_best_params_start_time = time()
                    space = model_builder().get_tuneable_params()
                    if parametric == True:
                        wf = model()
                        search = RandomizedSearchCV(wf, space, n_iter=N_ITER, cv=N_SPLITS, random_state=0)
                        x_ti_wf = pd.concat([ti_new[0].reset_index(drop=True),
                                            pd.DataFrame(ti_new[1]['Event'], columns=['Event'])], axis=1)
                        y_ti_wf = np.array([x[1] for x in ti_new[1]], float)
                        search.fit(x_ti_wf, y_ti_wf)
                    elif model_class_name == "XGBRegressor":
                        search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv= N_SPLITS, random_state=0)
                        y_ti_xgb = [x[1] if x[0] else -x[1] for x in ti_new[1]]
                        search.fit(ti_new[0], y_ti_xgb)
                    elif model_class_name == "DeepSurv":
                        search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv= N_SPLITS, random_state=0)
                        search.fit(ti_new_NN[0], ti_new_NN[1])                        
                    else:
                        search = RandomizedSearchCV(model, space, n_iter=N_ITER, cv= N_SPLITS, random_state=0)
                        search.fit(ti_new[0], ti_new[1])
                    best_params = search.best_params_
                    get_best_params_time = time() - get_best_params_start_time

                    # Train on train set TI with new params
                    model_train_start_time = time()
                    if parametric == True:
                        model = search.best_estimator_
                        model.fit(x_ti_wf, y_ti_wf)
                    elif model_class_name == "XGBRegressor":
                        model = search.best_estimator_
                        model.fit(ti_new[0], y_ti_xgb)
                    elif model_class_name == "DeepSurv":
                        model = search.best_estimator_
                        model.fit(ti_new_NN[0], ti_new_NN[1], val_NN)
                    else:
                        model = search.best_estimator_
                        model.fit(ti_new[0], ti_new[1])
                    model_train_time = time() - model_train_start_time

                    # Get C-index scores from current fold CVI
                    model_ci_inference_start_time = time()
                    if parametric == True:
                        x_cvi_wf = pd.concat([cvi_new[0].reset_index(drop=True),
                                              pd.DataFrame(cvi_new[1]['Event'],
                                                           columns=['Event'])], axis=1)
                        preds = model.predict(x_cvi_wf)
                        c_index = concordance_index(cvi[1]['Survival_time'], preds, cvi[1]['Event'])
                    else:
                        preds = model.predict(cvi_new[0])
                        c_index = concordance_index_censored(cvi[1]['Event'], cvi[1]['Survival_time'], preds)[0]
                    model_ci_inference_time = time() - model_ci_inference_start_time

                    # Get BS scores from current fold CVI
                    model_bs_inference_start_time = time()
                    if parametric == True:
                        model_instance = model.lifelines_model
                        # lower, upper = np.percentile(cvi_new[1][cvi_new[1].dtype.names[1]], [10, 90])
                        # times = np.arange(lower, upper+1)
                        surv_prob = model_instance.predict_survival_function(cvi_new[0]).T
                        brier_score = approx_brier_score(cvi_new[1], surv_prob)
                    elif model_class_name == "XGBRegressor" or model_class_name == "FastSurvivalSVM":
                        brier_score = np.nan
                    else:
                        # lower, upper = np.percentile(cvi_new[1][cvi_new[1].dtype.names[1]], [10, 90])
                        # times = np.arange(lower, upper)
                        surv_probs = pd.DataFrame(np.row_stack([fn(times)
                                                                for fn in model.predict_survival_function(cvi_new[0])]))
                        brier_score = approx_brier_score(cvi_new[1], surv_probs)
                    model_bs_inference_time = time() - model_bs_inference_start_time

                    t_total_split_time = time() - split_start_time
                    print(f"Evaluated {model_print_name} - {ft_selector_print_name}" + \
                          f" - CI={round(c_index, 3)} - BS={round(brier_score, 3)} - T={round(t_total_split_time, 3)}")

                    # Record results
                    res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, c_index, brier_score,
                                        get_best_features_time, get_best_params_time, model_train_time,
                                        model_ci_inference_time, model_bs_inference_time, t_total_split_time,
                                        best_params, selected_fts],
                                        index=["ModelName", "FtSelectorName", "NRepeat", "CIndex", "BrierScore",
                                               "TBestFeatures", "TBestParams", "TModelTrain",
                                               "TModelCIInference", "TModelBSInference", "TTotalSplit",
                                               "BestParams", "SelectedFts"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)

        # Save model results
        file_name = f"{model_name}_results.csv"
        model_results.to_csv("data/logs/" + file_name)

if __name__ == "__main__":
    main()