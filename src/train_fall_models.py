import numpy as np
import pandas as pd
from tools import file_reader, file_writer
from pathlib import Path
import config as cfg
from xgbse.non_parametric import calculate_kaplan_vectorized
from sklearn.model_selection import train_test_split
from utility.survival import predict_hazard_function, predict_survival_funciton
import shap
from tools import regressors, feature_selectors
from xgbse.metrics import approx_brier_score
from sksurv.metrics import concordance_index_censored

def main():
    df = file_reader.read_csv(Path.joinpath(cfg.PROCESSED_DATA_DIR, 'home_care_ma.csv'))
    X = df.drop(['Observed', 'Weeks'], axis=1)
    y = np.array(list(tuple(x) for x in df[['Observed', 'Weeks']].to_numpy()),
                 dtype=[('Observed', 'bool'), ('Weeks', '<f8')])

    # Train 4 estimators + KM and get their surv preds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    lower, upper = np.percentile(y_test['Weeks'], [10, 90])
    time_bins = np.arange(int(lower), int(upper+1))
    
    cph_model = regressors.Cph.make_model(regressors.Cph().get_best_params())
    rsf_model = regressors.RSF.make_model(regressors.RSF().get_best_params())
    cb_model = regressors.CoxBoost.make_model(regressors.CoxBoost().get_best_params())
    xgb_model = regressors.XGBTree.make_model(regressors.XGBTree().get_best_params())
    
    best_features = feature_selectors.SelectKBest10(X_train, y_train, xgb_model).get_features()
    X_train, X_test = X_train.loc[:,best_features], X_test.loc[:,best_features]

    cph_model.fit(X_train, y_train)
    rsf_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)
    y_train_xgb = [x[1] if x[0] else -x[1] for x in y_train]
    xgb_model.fit(X_train, y_train_xgb)

    cph_surv_func = predict_survival_funciton(cph_model, X_test, y_test)
    rsf_surv_func = predict_survival_funciton(rsf_model, X_test, y_test)
    cb_surv_func = predict_survival_funciton(cb_model, X_test, y_test)

    cph_hazard_func = predict_hazard_function(cph_model, X_test, y_test)
    rsf_hazard_func = predict_hazard_function(rsf_model, X_test, y_test)
    cb_hazard_func = predict_hazard_function(cb_model, X_test, y_test)

    km_mean, km_high, km_low = calculate_kaplan_vectorized(y_test['Weeks'].reshape(1,-1),
                                                           y_test['Observed'].reshape(1,-1),
                                                           time_bins)
    
    # Calculate test results
    cph_c_index = concordance_index_censored(y_test['Observed'], y_test['Weeks'],
                                             cph_model.predict(X_test))[0]
    rsf_c_index = concordance_index_censored(y_test['Observed'], y_test['Weeks'],
                                             rsf_model.predict(X_test))[0]
    cb_c_index = concordance_index_censored(y_test['Observed'], y_test['Weeks'],
                                            cb_model.predict(X_test))[0]
    xgb_c_index = concordance_index_censored(y_test['Observed'], y_test['Weeks'],
                                             xgb_model.predict(X_test))[0]
    
    # CPH BS
    cph_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cph_model.predict_survival_function(X_test)]))
    cph_bs = approx_brier_score(y_test, cph_surv_probs)
    
    # RSF BS
    rsf_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in rsf_model.predict_survival_function(X_test)]))
    rsf_bs = approx_brier_score(y_test, rsf_surv_probs)
    
    # CB BS
    cb_surv_probs = pd.DataFrame(np.row_stack([fn(time_bins) for fn in cb_model.predict_survival_function(X_test)]))
    cb_bs = approx_brier_score(y_test, cb_surv_probs)
    
    print(f"CPH model: {cph_c_index}/{cph_bs}")
    print(f"RSF model: {rsf_c_index}/{rsf_bs}")
    print(f"CB model: {cb_c_index}/{cb_bs}")
    print(f"XGB model: {xgb_c_index}/NA")

    # Make SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    # Save data
    file_writer.write_pickle(Path.joinpath(cfg.REPORTS_DIR,'best_features_arr.pkl'), best_features)
    file_writer.write_pickle(Path.joinpath(cfg.REPORTS_DIR,'xgb_shap_values.csv'), shap_values)
    file_writer.write_pickle(Path.joinpath(cfg.REPORTS_DIR,'xgb_shap_interaction_values.csv'), shap_interaction_values)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'cph_surv_func.csv'), cph_surv_func)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'rsf_surv_func.csv'), rsf_surv_func)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'cb_surv_func.csv'), cb_surv_func)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'cph_hazard_func.csv'), cph_hazard_func)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'rsf_hazard_func.csv'), rsf_hazard_func)
    file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR,'cb_hazard_func.csv'), cb_hazard_func)
    file_writer.write_km_surv_preds(Path.joinpath(cfg.REPORTS_DIR,'km_preds.csv'), km_mean, km_high, km_low)

if __name__ == "__main__":
    main()