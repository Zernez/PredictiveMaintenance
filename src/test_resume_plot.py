import numpy as np
import pandas as pd
import math
import shap
from pathlib import Path
import config as cfg
from xgbse.non_parametric import calculate_kaplan_vectorized
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools import regressors, feature_selectors
from xgbse.metrics import approx_brier_score
from sksurv.metrics import concordance_index_censored
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from auton_survival.metrics import survival_regression_metric
from auton_survival import DeepCoxPH


N_BOOT = 2

def main():

    data_util = DataETL()
    survival= Survival()

    cov, boot, info_pack = FileReader().read_data_xjtu()
    X, y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)
    S1, S2 = (X_train, y_train), (X_test, y_test)

    set_tr, set_te, set_tr_NN, set_te_NN = data_util.format_main_data(S1, S2)
    set_tr, set_te, set_tr_NN, set_te_NN = data_util.centering_main_data(set_tr, set_te, set_tr_NN, set_te_NN)

    X_train = set_tr [0]
    y_train = set_tr [1]
    X_test = set_te [0]
    y_test = set_te[1]
    X_train_NN = set_tr_NN [0]
    y_train_NN = set_tr_NN [1]
    X_test_NN = set_te_NN [0]
    y_test_NN = set_te_NN [1]


    X_test_NN, y_test_NN = data_util.control_censored_data(X_test_NN, y_test_NN, percentage= 10)
 
    lower, upper = np.percentile(y['Survival_time'], [10, 90])
    time_bins = np.arange(int(lower), int(upper))
    lower_NN, upper_NN = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
    times = np.arange(np.ceil(lower_NN), np.floor(upper_NN)).tolist()

    cph_model = regressors.Cph.make_model(regressors.Cph().get_best_params())
    cb_model = regressors.CoxBoost.make_model(regressors.CoxBoost().get_best_params())
    rsf_model = regressors.RSF.make_model(regressors.RSF().get_best_params())
    xgb_model = regressors.XGBTree.make_model(regressors.XGBTree().get_best_params())
    boost_model = regressors.GradientBoosting.make_model(regressors.GradientBoosting().get_best_params())
    boostD_model = regressors.GradientBoostingDART.make_model(regressors.GradientBoostingDART().get_best_params())
    SVM_model = regressors.SVM.make_model(regressors.SVM().get_best_params())
    NN_model = regressors.DeepSurv().make_model()
    NN_params= regressors.DeepSurv().get_best_hyperparams()

    best_features = feature_selectors.SelectKBest4(X_train, y_train, xgb_model).get_features()
    X_train, X_test, X_train_NN, X_test_NN = X_train.loc[:,best_features], X_test.loc[:,best_features], X_train_NN.loc[:,best_features], X_test_NN.loc[:,best_features]


    x= X_train_NN.to_numpy()
    t= y_train_NN.loc[:,"time"].to_numpy()
    e= y_train_NN.loc[:,"event"].to_numpy()

    cph_model.fit(X_train, y_train)
    rsf_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)
    y_train_xgb = [x[1] if x[0] else -x[1] for x in y_train]
    xgb_model.fit(X_train, y_train_xgb)
    boost_model.fit(X_train, y_train)
    boostD_model.fit(X_train, y_train)
    SVM_model.fit(X_train, y_train)
    NN_model.fit(x, t, e, vsize=0.2, iters= 40, **NN_params)

    cph_surv_func = survival.predict_survival_function(cph_model, X_test, y_test, lower, upper)
    rsf_surv_func = survival.predict_survival_function(rsf_model, X_test, y_test, lower, upper)
    cb_surv_func = survival.predict_survival_function(cb_model, X_test, y_test, lower, upper)
    boost_surv_func = survival.predict_survival_function(boost_model, X_test, y_test, lower, upper)
    boostD_surv_func = survival.predict_survival_function(boostD_model, X_test, y_test, lower, upper)
    x= X_test_NN.to_numpy()
    NN_surv_func = NN_model.predict_survival(x, times)

    cph_hazard_func = survival.predict_hazard_function(cph_model, X_test, y_test, lower, upper)
    rsf_hazard_func = survival.predict_hazard_function(rsf_model, X_test, y_test, lower, upper)
    cb_hazard_func = survival.predict_hazard_function(cb_model, X_test, y_test, lower, upper)
    boost_hazard_func = survival.predict_hazard_function(boost_model, X_test, y_test, lower, upper)
    boostD_hzazard_func = survival.predict_hazard_function(boostD_model, X_test, y_test, lower, upper)
    NN_hazard_func = NN_model.predict_risk(x, times)

    km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Event'].reshape(1,-1),
                                                           y_test['Survival_time'].reshape(1,-1),
                                                           time_bins)

    # Calculate test results
    cph_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             cph_model.predict(X_test))[0]
    rsf_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             rsf_model.predict(X_test))[0]
    cb_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                            cb_model.predict(X_test))[0]
    xgb_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             xgb_model.predict(X_test))[0]
    boost_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             boost_model.predict(X_test))[0]
    boostD_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             boostD_model.predict(X_test))[0]
    SVM_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                             SVM_model.predict(X_test))[0]
    NN_c_index = np.mean(survival_regression_metric('ctd', y_test_NN,
                                                    NN_surv_func,
                                                    times=times))

    # CPH BS
    cph_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cph_model.predict_survival_function(X_test)]))
    cph_bs = approx_brier_score(y_test, cph_surv_probs)

    # RSF BS
    rsf_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in rsf_model.predict_survival_function(X_test)]))
    rsf_bs = approx_brier_score(y_test, rsf_surv_probs)

    # CB BS
    cb_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cb_model.predict_survival_function(X_test)]))
    cb_bs = approx_brier_score(y_test, cb_surv_probs)

    # Boost BS
    boost_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in boost_model.predict_survival_function(X_test)]))
    boost_bs = approx_brier_score(y_test, boost_surv_probs)

    # BoostD BS
    boostD_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in boostD_model.predict_survival_function(X_test)]))
    boostD_bs = approx_brier_score(y_test, boostD_surv_probs)

    # NN BS
    NN_bs = np.mean(survival_regression_metric('brs', y_test_NN,
                                                    NN_surv_func,
                                                    times=times))

    print(f"CPH model: {cph_c_index}/{cph_bs}")
    print(f"RSF model: {rsf_c_index}/{rsf_bs}")
    print(f"CB model: {cb_c_index}/{cb_bs}")
    print(f"XGB model: {xgb_c_index}/NA")
    print(f"Boost model: {boost_c_index}/{boost_bs}")
    print(f"BoostD model: {boostD_c_index}/{boostD_bs}")
    print(f"SVM model: {SVM_c_index}/NA")
    print(f"NN model: {NN_c_index}/{NN_bs}")

    shap_pack= []

    # Make SHAP values

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    explainer_CPH = shap.Explainer(cph_model.predict, X_test)
    shap_values_CPH = explainer_CPH(X_test)

    explainer_CB = shap.Explainer(cb_model.predict, X_test)
    shap_values_CB = explainer_CB(X_test)

    explainer_RSF = shap.Explainer(rsf_model.predict, X_test)
    shap_values_RSF = explainer_RSF(X_test)

    explainer_B = shap.Explainer(boost_model.predict, X_test)
    shap_values_B = explainer_B(X_test)

    explainer_BD = shap.Explainer(boostD_model.predict, X_test)
    shap_values_BD = explainer_BD(X_test)

    explainer_NN = shap.Explainer(NN_model.predict_survival, X_test)
    shap_values_NN = explainer_NN(X_test)

    shap.plots.waterfall(shap_values_CB[0])
    shap.plots.beeswarm(shap_values_CB)

if __name__ == "__main__":
    main()