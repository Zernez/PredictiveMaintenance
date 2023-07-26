import numpy as np
import pandas as pd
import random
import shap
from pathlib import Path
import config as cfg
from xgbse.non_parametric import calculate_kaplan_vectorized
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.resume import Resume
from tools import regressors, feature_selectors
from utility.builder import Builder
from xgbse.metrics import approx_brier_score
from sksurv.metrics import concordance_index_censored
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from auton_survival.metrics import survival_regression_metric
from lifelines import WeibullAFTFitter

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# N_BOOT = 3
# BEARINGS= 2
# BOOT_NO= 200
# N_REPEATS = 5
# NEW_DATASET = False
# DATASET = "pronostia"
# TEST_SIZE= 0.3

N_BOOT = 3
BEARINGS= 5
BOOT_NO= 200
N_REPEATS = 5
NEW_DATASET = False
DATASET = "xjtu"
TEST_SIZE= 0.3

def main():

    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap= N_BOOT)     

    data_util = DataETL(DATASET)
    survival= Survival()

    cov, boot, info_pack = FileReader(DATASET).read_data()
    X, y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT)

    Resumer = Resume(X, y, DATASET)

    Resumer.table_result_hyper()

    Resumer.presentation(BEARINGS, BOOT_NO)
    df_CI = pd.DataFrame(columns= ["Model", "CI score"])
    df_B = pd.DataFrame(columns= ["Model", "Brier score"])

    for n_repeat in range(N_REPEATS):

        seed= random.randint(0,30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SIZE, random_state= seed)
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


#        X_test_NN, y_test_NN = data_util.control_censored_data(X_test_NN, y_test_NN, percentage= 10)
    
        # lower, upper = np.percentile(y['Survival_time'], [10, 90])
        # time_bins = np.arange(int(lower + 1), int(upper - 1))
        # lower_NN, upper_NN = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        # times = np.arange(np.ceil(lower_NN), np.floor(upper_NN)).tolist()

        weibull_model = WeibullAFTFitter(alpha= 0.2, penalizer= 0.02)
        cph_model = regressors.Cph.make_model(regressors.Cph().get_best_params())
        cphLASSO_model = regressors.CphLASSO.make_model(regressors.CphLASSO().get_best_params())
        rsf_model = regressors.RSF.make_model(regressors.RSF().get_best_params())
        boost_model = regressors.GradientBoosting.make_model(regressors.GradientBoosting().get_best_params())
        boostDART_model = regressors.GradientBoostingDART.make_model(regressors.GradientBoostingDART().get_best_params())
        SVM_model = regressors.SVM.make_model(regressors.SVM().get_best_params())
        NN_model = regressors.DeepSurv().make_model()
        NN_params= regressors.DeepSurv().get_best_hyperparams()

        best_features = feature_selectors.SelectKBest4(X_train, y_train, cph_model).get_features()
        X_train, X_test, X_train_NN, X_test_NN = X_train.loc[:,best_features], X_test.loc[:,best_features], X_train_NN.loc[:,best_features], X_test_NN.loc[:,best_features]

        lower_NN, upper_NN = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        times = np.arange(np.ceil(lower_NN + 2), np.floor(upper_NN - 1)).tolist()
        lower, upper = np.percentile(y_train[y_test.dtype.names[1]], [10, 90])
        time_bins = np.arange(np.ceil(lower + 2), np.floor(upper - 1)).tolist()

        x= X_train_NN.to_numpy()
        t= y_train_NN.loc[:,"time"].to_numpy()
        e= y_train_NN.loc[:,"event"].to_numpy()

        X_train_WB = pd.concat([X_train.reset_index(drop=True),
                                pd.DataFrame(y_train['Survival_time'], columns=['Survival_time'])], axis=1)
        X_train_WB = pd.concat([X_train_WB.reset_index(drop=True),
                        pd.DataFrame(y_train['Event'], columns=['Event'])], axis=1)
        X_test_WB = pd.concat([X_test.reset_index(drop=True),
                                pd.DataFrame(y_test['Survival_time'], columns=['Survival_time'])], axis=1)
        X_test_WB = pd.concat([X_test_WB.reset_index(drop=True),
                        pd.DataFrame(y_test['Event'], columns=['Event'])], axis=1)

        weibull_model.fit(X_train_WB, duration_col='Survival_time', event_col='Event')
        cph_model.fit(X_train, y_train)
        cphLASSO_model.fit(X_train, y_train)
        rsf_model.fit(X_train, y_train)
        boost_model.fit(X_train, y_train)
        boostDART_model.fit(X_train, y_train)
        SVM_model.fit(X_train, y_train)
        NN_model.fit(x, t, e, vsize=0.3, iters= 80, **NN_params)

        weibull_surv_func = survival.predict_survival_function(weibull_model, X_test_WB, y_test, lower, upper)
        cph_surv_func = survival.predict_survival_function(cph_model, X_test, y_test, lower, upper)
        cphLASSO_surv_func = survival.predict_survival_function(cphLASSO_model, X_test, y_test, lower, upper)
        boost_surv_func = survival.predict_survival_function(boost_model, X_test, y_test, lower, upper)    
        boostDART_surv_func = survival.predict_survival_function(boostDART_model, X_test, y_test, lower, upper)
        NN_surv_func = survival.predict_survival_function(NN_model, X_test_NN, y_test, times, times)
        rsf_surv_func = survival.predict_survival_function(rsf_model, X_test, y_test, lower, upper)

        weibull_hazard_func = survival.predict_hazard_function(weibull_model, X_test_WB, y_test, lower, upper)
        cph_hazard_func = survival.predict_hazard_function(cph_model, X_test, y_test, lower, upper)
        cphLASSO_hazard_func = survival.predict_hazard_function(cphLASSO_model, X_test, y_test, lower, upper)   
        rsf_hazard_func = survival.predict_hazard_function(rsf_model, X_test, y_test, lower, upper)
        boost_hazard_func = survival.predict_hazard_function(boost_model, X_test, y_test, lower, upper)
        boostDART_hzazard_func = survival.predict_hazard_function(boostDART_model, X_test, y_test, lower, upper)

        km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Event'].reshape(1,-1),
                                                            y_test['Survival_time'].reshape(1,-1),
                                                            time_bins)

        # WEIBULL C_INDEX
        weibull_c_index = np.mean(weibull_model.concordance_index_)  

        # CPH C_INDEX
        cph_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                cph_model.predict(X_test))[0]
        
        # CPH LASSO C_INDEX        
        cphLASSO_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                cphLASSO_model.predict(X_test))[0]
        
        # RSF C_INDEX        
        rsf_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                rsf_model.predict(X_test))[0]
        
        # Boost C_INDEX        
        boost_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                boost_model.predict(X_test))[0]
        
        # Boost DART C_INDEX        
        boostDART_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                boostDART_model.predict(X_test))[0]
        
        # SVM C_INDEX        
        SVM_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                SVM_model.predict(X_test))[0]
        
        # NN C_INDEX        
        NN_c_index = np.mean(survival_regression_metric('ctd', y_test_NN,
                                                        NN_surv_func,
                                                        times=times))
        
        # WEIBULL BS
        weibull_surv_probs = weibull_model.predict_survival_function(X_test_WB).T
        weibull_bs = approx_brier_score(y_test, weibull_surv_probs)

        # CPH BS
        cph_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cph_model.predict_survival_function(X_test)]))
        cph_bs = approx_brier_score(y_test, cph_surv_probs)

        # CPH LASSO BS
        cphLASSO_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cphLASSO_model.predict_survival_function(X_test)]))
        cphLASSO_bs = approx_brier_score(y_test, cphLASSO_surv_probs)

        # RSF BS
        rsf_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in rsf_model.predict_survival_function(X_test)]))
        rsf_bs = approx_brier_score(y_test, rsf_surv_probs)

        # Boost BS
        boost_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in boost_model.predict_survival_function(X_test)]))
        boost_bs = approx_brier_score(y_test, boost_surv_probs)

        # Boost DART BS
        boostDART_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in boostDART_model.predict_survival_function(X_test)]))
        boostDART_bs = approx_brier_score(y_test, boostDART_surv_probs)

        # NN BS
        NN_bs = np.mean(survival_regression_metric('brs', y_test_NN,
                                                        NN_surv_func,
                                                        times=times))

        print(f"Weibull model: {weibull_c_index}/{weibull_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Weibull AFT", weibull_c_index, weibull_bs)
        print(f"CPH model: {cph_c_index}/{cph_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Cox PH", cph_c_index, cph_bs)
        print(f"CPH LASSO model: {cphLASSO_c_index}/{cphLASSO_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Cox PH", cphLASSO_c_index, cphLASSO_bs)
        print(f"RSF model: {rsf_c_index}/{rsf_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Random Survival Forest", rsf_c_index, rsf_bs)
        print(f"Boost model: {boost_c_index}/{boost_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Gradient Boosting", boost_c_index, boost_bs)
        print(f"Boost DART model: {boostDART_c_index}/{boostDART_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Gradient Boosting DART", boostDART_c_index, boostDART_bs)
        print(f"SVM model: {SVM_c_index}/NA")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "SVM", SVM_c_index, None)
        print(f"NN model: {NN_c_index}/{NN_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Neural Network", NN_c_index, NN_bs)

        # Make SHAP values
        if (n_repeat== N_REPEATS - 1):

            explainer_CPH = shap.Explainer(cph_model.predict, X_test)
            shap_values_CPH = explainer_CPH(X_test)

            explainer_CPHLASSO = shap.Explainer(cphLASSO_model.predict, X_test)
            shap_values_CPHLASSO = explainer_CPHLASSO(X_test)

            explainer_RSF = shap.Explainer(rsf_model.predict, X_test)
            shap_values_RSF = explainer_RSF(X_test)

            explainer_B = shap.Explainer(boost_model.predict, X_test)
            shap_values_B = explainer_B(X_test)

            explainer_BD = shap.Explainer(boostDART_model.predict, X_test)
            shap_values_BD = explainer_BD(X_test)

            explainer_SVM = shap.Explainer(SVM_model.predict, X_test)
            shap_values_SVM = explainer_SVM(X_test)

            explainer_NN = shap.Explainer(NN_model.predict_survival, X_test_NN)
            shap_values_NN = explainer_NN(X_test)
    
    Resumer.plot_simple_sl (y_test, weibull_surv_probs, "Weibull AFT")

    Resumer.plot_simple_sl (y_test, cph_surv_func, "Cox PH")
    Resumer.plot_sl_ci (y_test, cph_surv_func, "Cox PH")
    Resumer.plot_shap(explainer_CPH, shap_values_CPH, X_test, "Cox PH")

    Resumer.plot_simple_sl (y_test, cphLASSO_surv_func, "Cox PH LASSO")
    Resumer.plot_sl_ci (y_test, cphLASSO_surv_func, "Cox PH LASSO")
    Resumer.plot_shap(explainer_CPHLASSO, shap_values_CPHLASSO, X_test, "Cox PH LASSO")

    Resumer.plot_simple_sl (y_test, rsf_surv_func, "Random Survival Forest")
    Resumer.plot_sl_ci (y_test, rsf_surv_func, "Random Survival Forest")
    Resumer.plot_shap(explainer_RSF, shap_values_RSF, X_test, "Random Survival Forest")

    Resumer.plot_simple_sl (y_test, boost_surv_func, "Gradient Boosting")
    Resumer.plot_sl_ci (y_test, boost_surv_func, "Gradient Boosting")
    Resumer.plot_shap(explainer_B, shap_values_B, X_test, "Gradient Boosting")

    Resumer.plot_simple_sl (y_test, boostDART_surv_func, "Gradient Boosting DART")
    Resumer.plot_sl_ci (y_test, boostDART_surv_func, "Gradient Boosting DART")
    Resumer.plot_shap(explainer_BD, shap_values_BD, X_test, "Gradient Boosting DART")

    Resumer.plot_shap(explainer_SVM, shap_values_SVM, X_test, "Support Vector Machine") 
 
    NN_surv_func = pd.DataFrame(NN_surv_func)
    Resumer.plot_simple_sl (y_test, NN_surv_func, "Neural Network")
#    Resumer.plot_shap(explainer_NN, shap_values_NN, X_test, "Neural Network")

    df_CI, df_B = Resumer.plot_performance(True, df_CI, df_B)

if __name__ == "__main__":
    main()