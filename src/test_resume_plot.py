import numpy as np
import pandas as pd
import random
#import shap
from pathlib import Path
import config as cfg
from xgbse.non_parametric import calculate_kaplan_vectorized
from sklearn.model_selection import train_test_split
from utility.survival import Survival
from tools.resume import Resume
from tools import regressors, feature_selectors
from utility.builder import Builder
from xgbse.metrics import approx_brier_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sklearn.model_selection import KFold
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from auton_survival.metrics import survival_regression_metric
from lifelines import WeibullAFTFitter
from lifelines import KaplanMeierFitter
import time

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = 3
BEARINGS= 2
BOOT_NO= 200
N_REPEATS = 1
NEW_DATASET = False
DATASET = "pronostia"
TEST_SIZE= 0.3
TYPE= "correlated" # not_correlated
LINE_PLOT= 3
FEATURE_TO_SPLIT= "rms"
SPLIT_THRESHOLD= []
SPLITTED = True

def main():

    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap= N_BOOT)     

    data_util = DataETL(DATASET)
    survival= Survival()

    cov, boot, info_pack = FileReader(DATASET).read_data()
    X, y = data_util.make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)

    Resumer = Resume(X, y, DATASET)

#    Resumer.table_result_hyper()
    
#    Resumer.presentation(BEARINGS, BOOT_NO)
    df_CI = pd.DataFrame(columns= ["Model", "CI score"])
    df_B = pd.DataFrame(columns= ["Model", "Brier score"])

    for n_repeat in range(N_REPEATS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= TEST_SIZE, random_state = 0)
        S1, S2 = (X_train, y_train), (X_test, y_test)

        set_tr, set_te, set_tr_NN, set_te_NN = data_util.format_main_data(S1, S2)
        percent_ref= data_util.calculate_positions_percentages(set_te[0], FEATURE_TO_SPLIT, SPLIT_THRESHOLD)
        set_tr, set_te, set_tr_NN, set_te_NN = data_util.centering_main_data(set_tr, set_te, set_tr_NN, set_te_NN)
        val_ref= data_util.find_values_by_percentages(set_te[0], FEATURE_TO_SPLIT, percent_ref)

        X_train = set_tr [0]
        y_train = set_tr [1]
        X_test = set_te [0]
        y_test = set_te [1]
        X_train_NN = set_tr_NN [0]
        y_train_NN = set_tr_NN [1]
        X_test_NN = set_te_NN [0]
        y_test_NN = set_te_NN [1]

#        X_test_NN, y_test_NN = data_util.control_censored_data(X_test_NN, y_test_NN, percentage= 10)
    
        lower, upper = np.percentile(y['Survival_time'], [10, 90])
        time_bins = np.arange(np.ceil(lower), np.floor(upper))
        lower_NN, upper_NN = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        times = np.arange(np.ceil(lower_NN), np.floor(upper_NN))

        weibull_model = WeibullAFTFitter(alpha= 0.2, penalizer= 0.02)
        cph_model = regressors.CoxPH.make_model(regressors.CoxPH().get_best_params())
        rsf_model = regressors.RSF.make_model(regressors.RSF().get_best_params())
        boost_model = regressors.CoxBoost.make_model(regressors.CoxBoost().get_best_params())
        NN_model = regressors.DeepSurv().make_model()
        NN_params= regressors.DeepSurv().get_best_hyperparams()
        DSM_model = regressors.DSM().make_model()
        DSM_params= regressors.DSM().get_best_hyperparams()

        best_features = feature_selectors.NoneSelector(X_train, y_train, cph_model).get_features()
        X_train, X_test, X_train_NN, X_test_NN = X_train.loc[:,best_features], X_test.loc[:,best_features], X_train_NN.loc[:,best_features], X_test_NN.loc[:,best_features]

        x= X_train_NN.to_numpy()
        t= y_train_NN['time'].to_numpy()
        e= y_train_NN['event'].to_numpy()
        xte= X_test_NN.to_numpy()

        X_train_WB = pd.concat([X_train.reset_index(drop=True),
                                pd.DataFrame(y_train['Survival_time'], columns=['Survival_time'])], axis=1)
        X_train_WB = pd.concat([X_train_WB.reset_index(drop=True),
                        pd.DataFrame(y_train['Event'], columns=['Event'])], axis=1)
        X_test_WB = pd.concat([X_test.reset_index(drop=True),
                                pd.DataFrame(y_test['Survival_time'], columns=['Survival_time'])], axis=1)
        X_test_WB = pd.concat([X_test_WB.reset_index(drop=True),
                        pd.DataFrame(y_test['Event'], columns=['Event'])], axis=1)
        
        start_time_weibull = time.time()
        weibull_model.fit(X_train_WB, duration_col='Survival_time', event_col='Event')
        end_time_weibull =time.time()-start_time_weibull
        start_time_cph = time.time()
        cph_model.fit(X_train, y_train)
        end_time_cph = time.time()-start_time_cph
        start_time_rsf = time.time()
        rsf_model.fit(X_train, y_train)
        end_time_rsf = time.time()-start_time_rsf
        start_time_boost = time.time()
        boost_model.fit(X_train, y_train)
        end_time_boost = time.time()-start_time_boost
        start_time_NN = time.time()
        NN_model.fit(x, t, e, vsize=0.3, **NN_params)
        end_time_NN = time.time()-start_time_NN
        start_time_DSM = time.time()
        DSM_model.fit(x, t, e, vsize=0.3, **DSM_params)
        end_time_DSM = time.time()-start_time_DSM

        lower_NN_tr, upper_NN_tr = np.percentile(y_train[y_train.dtype.names[1]], [10, 90])
        times_tr = np.arange(np.ceil(lower_NN_tr), np.floor(upper_NN_tr))

        weibull_surv_func = survival.predict_survival_function(weibull_model, X_test_WB, time_bins)
        cph_surv_func = survival.predict_survival_function(cph_model, X_test, time_bins)
        boost_surv_func = survival.predict_survival_function(boost_model, X_test, time_bins)    
        rsf_surv_func = survival.predict_survival_function(rsf_model, X_test, time_bins)
        NN_surv_func = survival.predict_survival_function(NN_model, xte, max(times))
        NN_surv_func_tr = survival.predict_survival_function(NN_model, x, max(times_tr))
        DSM_surv_func = survival.predict_survival_function(DSM_model, xte, max(times))
        DSM_surv_func_tr = survival.predict_survival_function(DSM_model, x, max(times_tr))

        weibull_hazard_func = survival.predict_hazard_function(weibull_model, X_test_WB, time_bins)
        cph_hazard_func = survival.predict_hazard_function(cph_model, X_test, time_bins)
        boost_hazard_func = survival.predict_hazard_function(boost_model, X_test, time_bins)
        rsf_hazard_func = survival.predict_hazard_function(rsf_model, X_test, time_bins)


        km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Event'].reshape(1,-1),
                                                            y_test['Survival_time'].reshape(1,-1),
                                                            time_bins)

        # WEIBULL C_INDEX
        weibull_c_index = np.mean(weibull_model.concordance_index_)
        y_test_td= y_test
        size= y_test.size
        temp_pred = weibull_model.predict_survival_function(X_test_WB)[0]
        sub= size - len(temp_pred)
        if (sub < 0):
            temp_pred.drop(temp_pred.tail(-sub).index,inplace=True)
        elif (sub > 0):
            y_test_td= y_test[:-sub]
            
        weibull_c_index_td = concordance_index_ipcw(y_train, y_test_td, temp_pred)[0]

        # CPH C_INDEX
        cph_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                cph_model.predict(X_test))[0]
        cph_c_index_td = concordance_index_ipcw(y_train, y_test,
                                                cph_model.predict(X_test))[0]
        
        # RSF C_INDEX        
        rsf_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                rsf_model.predict(X_test))[0]
        rsf_c_index_td = concordance_index_ipcw(y_train, y_test,
                                                rsf_model.predict(X_test))[0]
        
        # Boost C_INDEX        
        boost_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'],
                                                boost_model.predict(X_test))[0]
        boost_c_index_td = concordance_index_ipcw(y_train, y_test,
                                                boost_model.predict(X_test))[0]
        
        # NN C_INDEX        
        NN_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], NN_surv_func)[0]
        NN_c_index_tr = concordance_index_censored(y_train['Event'], y_train['Survival_time'], NN_surv_func_tr)[0]
        NN_c_index_td = concordance_index_ipcw(y_train, y_test, NN_surv_func)[0]

        # DSM C_INDEX        
        DSM_c_index = concordance_index_censored(y_test['Event'], y_test['Survival_time'], DSM_surv_func)[0]
        DSM_c_index_tr = concordance_index_censored(y_train['Event'], y_train['Survival_time'], DSM_surv_func_tr)[0]
        DSM_c_index_td = concordance_index_ipcw(y_train, y_test, DSM_surv_func)[0]


        # WEIBULL BS
        weibull_bs = approx_brier_score(y_test, weibull_surv_func)

        # CPH BS
        cph_surv_probs = pd.DataFrame(cph_surv_func)
        cph_bs = approx_brier_score(y_test, cph_surv_probs)

        # RSF BS
        rsf_surv_probs = pd.DataFrame(boost_surv_func)
        rsf_bs = approx_brier_score(y_test, rsf_surv_probs)

        # Boost BS
        boost_surv_probs = pd.DataFrame(rsf_surv_func)
        boost_bs = approx_brier_score(y_test, boost_surv_probs)

        # NN BS
        NN_surv_probs = NN_model.predict_survival(xte)
        # NN_surv_probs = pd.DataFrame(np.row_stack([NN_model.predict_risk(xte, t= time).flatten() for time in times]).T)
        NN_bs = approx_brier_score(y_test, NN_surv_probs)

        # NN_surv_probs_tr = pd.DataFrame(np.row_stack([NN_model.predict_risk(x, t= time).flatten() for time in times_tr]).T)
        NN_surv_probs_tr = NN_model.predict_survival(x)
        NN_bs_tr = approx_brier_score(y_train, NN_surv_probs_tr)

        # DSM BS
        # DSM_surv_probs = pd.DataFrame(np.row_stack([DSM_model.predict_risk(xte, t= time).flatten() for time in times]).T)
        DSM_surv_probs = pd.DataFrame(DSM_model.predict_survival(xte, t=times))
        DSM_bs = approx_brier_score(y_test, DSM_surv_probs)

        # DSM_surv_probs_tr = pd.DataFrame(np.row_stack([DSM_model.predict_risk(x, t= time).flatten() for time in times_tr]).T)
        DSM_surv_probs_tr = pd.DataFrame(NN_model.predict_survival(x, t= max(times_tr)))
        DSM_bs_tr = approx_brier_score(y_train, DSM_surv_probs_tr)


        print ("Performance info: ") 
        print(f"Weibull model: TrTime, CI, CItd, BS - {end_time_weibull}/{weibull_c_index}/{weibull_c_index_td}/{weibull_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Weibull AFT", weibull_c_index, weibull_bs)
        print(f"CPH model: TrTime, CI, CItd, BS - {end_time_cph}/{cph_c_index}/{cph_c_index_td}/{cph_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Cox PH", cph_c_index, cph_bs)
        print(f"RSF model: TrTime, CI, CItd, BS - {end_time_rsf}/{rsf_c_index}/{rsf_c_index_td}/{rsf_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Random Survival Forest", rsf_c_index, rsf_bs)
        print(f"Boost model: TrTime, CI, CItd, BS - {end_time_boost}/{boost_c_index}/{boost_c_index_td}/{boost_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Gradient Boosting", boost_c_index, boost_bs)
        print(f"NN model: TrTime, CI, CItd, BS - {end_time_NN}/{NN_c_index}/{NN_c_index_td}/{NN_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "Neural Network", NN_c_index, NN_bs)
        print(f"DSM model: TrTime, CI, CItd, BS - {end_time_DSM}/{DSM_c_index}/{DSM_c_index_td}/{DSM_bs}")
        df_CI, df_B = Resumer.plot_performance(False, df_CI, df_B, "DSM", DSM_c_index, DSM_bs)

        print ("Overfitting - underfitting info: ") 
        print(f"NN model train: TrCI, TrBS - {NN_bs}/{NN_bs_tr}")
        print(f"DSM model train: TrCI, TrBS - {DSM_bs}/{DSM_bs_tr}")

        print ("Latex performance table: ")
        print (f"CoxPH & {round(end_time_cph, 3)} & {round(cph_c_index, 3)} & {round(cph_c_index_td, 3)} & {round(cph_bs, 3)} \\\\" +
                f"RSF & {round(end_time_rsf, 3)} & {round(rsf_c_index, 3)} & {round(rsf_c_index_td, 3)} & {round(rsf_bs, 3)} \\\\" +
                f"CoxBoost & {round(end_time_boost, 3)} & {round(boost_c_index, 3)} & {round(boost_c_index_td, 3)} & {round(boost_bs, 3)}\\\\" +
                f"DeepSurv & {round(end_time_NN, 3)} & {round(NN_c_index, 3)} & {round(NN_c_index_td, 3)} & {round(NN_bs, 3)}\\\\" +       
                f"DSM & {round(end_time_DSM, 3)} & {round(DSM_c_index, 3)} & {round(DSM_c_index_td, 3)} & {round(DSM_bs, 3)}\\\\" +
                f"WeibullAFT & {round(end_time_weibull, 3)} & {round(weibull_c_index, 3)} & {round(weibull_c_index_td, 3)} & {round(weibull_bs, 3)}\\\\")
        
        #Plotting aggregate
        Km= KaplanMeierFitter()
        Km.fit(durations= X_test_WB["Survival_time"], event_observed= X_test_WB["Survival_time"])
        Km.predict(time_bins)
        
        surv_label= []
        surv_label.append(weibull_surv_func)
        surv_label.append(cph_surv_func)
        surv_label.append(boost_surv_func)
        surv_label.append(rsf_surv_func)
        surv_label.append(NN_surv_probs)
        surv_label.append(DSM_surv_probs)

        Resumer.plot_aggregate_sl (Km, surv_label)

        # # Make SHAP values
        # if (n_repeat== N_REPEATS - 1):

        #     explainer_CPH = shap.Explainer(cph_model.predict, X_test)
        #     shap_values_CPH = explainer_CPH(X_test)

        #     explainer_RSF = shap.Explainer(rsf_model.predict, X_test)
        #     shap_values_RSF = explainer_RSF(X_test)

        #     explainer_B = shap.Explainer(boost_model.predict, X_test)
        #     shap_values_B = explainer_B(X_test)

        #     explainer_NN = shap.Explainer(NN_model.predict_survival, X_test_NN)
        #     shap_values_NN = explainer_NN(X_test)
    
#     Resumer.plot_simple_sl (y_test, weibull_surv_func, "Weibull AFT")

#     Resumer.plot_simple_sl (y_test, cph_surv_func, "Cox PH")
#     Resumer.plot_sl_ci (y_test, cph_surv_func, "Cox PH")
#     # Resumer.plot_shap(explainer_CPH, shap_values_CPH, X_test, "Cox PH")

#     Resumer.plot_simple_sl (y_test, rsf_surv_func, "Random Survival Forest")
#     Resumer.plot_sl_ci (y_test, rsf_surv_func, "Random Survival Forest")
#     # Resumer.plot_shap(explainer_RSF, shap_values_RSF, X_test, "Random Survival Forest")

#     Resumer.plot_simple_sl (y_test, boost_surv_func, "Gradient Boosting")
#     Resumer.plot_sl_ci (y_test, boost_surv_func, "Gradient Boosting")
#     # Resumer.plot_shap(explainer_B, shap_values_B, X_test, "Gradient Boosting")
 
#     NN_surv_func = pd.DataFrame(NN_surv_func)
#     Resumer.plot_simple_sl (y_test, NN_surv_func, "Neural Network")
# #    Resumer.plot_shap(explainer_NN, shap_values_NN, X_test, "Neural Network")

#    df_CI, df_B = Resumer.plot_performance(True, df_CI, df_B)

if __name__ == "__main__":
    main()
