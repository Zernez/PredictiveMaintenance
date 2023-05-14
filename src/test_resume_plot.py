import numpy as np
import pandas as pd
import shap
from tools import file_reader, file_writer
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

N_BOOT = 2
N_MODELS= 7 #No NN

def main():

    cov, boot, info_pack = FileReader().read_data_xjtu()
    X, y = DataETL().make_surv_data_sklS(cov, boot, info_pack, N_BOOT)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lower, upper = np.percentile(y['Survival_time'], [10, 90])
    time_bins = np.arange(int(lower), int(upper+1))

    cph_model = regressors.Cph.make_model(regressors.Cph().get_best_params())
    cb_model = regressors.CoxBoost.make_model(regressors.CoxBoost().get_best_params())
    rsf_model = regressors.RSF.make_model(regressors.RSF().get_best_params())
    xgb_model = regressors.XGBTree.make_model(regressors.XGBTree().get_best_params())
    boost_model = regressors.GradientBoosting.make_model(regressors.GradientBoosting().get_best_params())
    boostD_model = regressors.GradientBoostingDART.make_model(regressors.GradientBoostingDART().get_best_params())     
    SVM_model = regressors.SVM.make_model(regressors.SVM().get_best_params())

    n_feature= 4    
    NN_model = regressors.DeepSurv().make_model(n_feature)

    best_features = feature_selectors.SelectKBest4(X_train, y_train, xgb_model).get_features()
    X_train, X_test = X_train.loc[:,best_features], X_test.loc[:,best_features]

    cph_model.fit(X_train, y_train)
    rsf_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)
    y_train_xgb = [x[1] if x[0] else -x[1] for x in y_train]
    xgb_model.fit(X_train, y_train_xgb)
    boost_model.fit(X_train, y_train)
    boostD_model.fit(X_train, y_train)
    SVM_model.fit(X_train, y_train)    
#    NN_model.fit(X_train, y_train) ##look at regressors.py implementation

    cph_surv_func = Survival().predict_survival_function(cph_model, X_test, y_test, lower, upper)
    rsf_surv_func = Survival().predict_survival_function(rsf_model, X_test, y_test, lower, upper)
    cb_surv_func = Survival().predict_survival_function(cb_model, X_test, y_test, lower, upper)
    boost_surv_func = Survival().predict_survival_function(boost_model, X_test, y_test, lower, upper)
    boostD_surv_func = Survival().predict_survival_function(boostD_model, X_test, y_test, lower, upper)
#    NN_surv_func = Survival().predict_survival_function(NN_model, X_test, y_test, lower, upper)

    cph_hazard_func = Survival().predict_hazard_function(cph_model, X_test, y_test, lower, upper)
    rsf_hazard_func = Survival().predict_hazard_function(rsf_model, X_test, y_test, lower, upper)
    cb_hazard_func = Survival().predict_hazard_function(cb_model, X_test, y_test, lower, upper)
    boost_surv_func = Survival().predict_hazard_function(boost_model, X_test, y_test, lower, upper)
    boostD_surv_func = Survival().predict_hazard_function(boostD_model, X_test, y_test, lower, upper)
#    NN_surv_func = Survival().predict_hazard_function(NN_model, X_test, y_test, lower, upper)

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
#    NN_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
#                                             NN_model.predict_surv(X_test))[0]
    
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
#    NN_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in NN_model.predict_surv(X_test)]))
#    NN_bs = approx_brier_score(y_test, NN_surv_probs)
    
    print(f"CPH model: {cph_c_index}/{cph_bs}")
    print(f"RSF model: {rsf_c_index}/{rsf_bs}")
    print(f"CB model: {cb_c_index}/{cb_bs}")
    print(f"XGB model: {xgb_c_index}/NA")
    print(f"Boost model: {boost_c_index}/{boost_bs}")
    print(f"BoostD model: {boostD_c_index}/{boostD_bs}")
    print(f"SVM model: {SVM_c_index}/NA")
    # print(f"NN model: {NN_c_index}/{NN_bs}")

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

    # explainer_NN = shap.Explainer(boostD_model.model.predict_surv(), X_test)
    # shap_values_NN = explainer_NN(X_test)

    shap.plots.waterfall(shap_values_CB[0])
    shap.plots.beeswarm(shap_values_CB)

if __name__ == "__main__":
    main()