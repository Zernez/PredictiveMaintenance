from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tools import regressors
from tools import file_writer
from pathlib import Path
import config as cfg
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest10, SelectKBest20, \
                                    RegMRMR10, RegMRMR20, RFE10, RFE20
from tools.regressors import Cph, CphRidge, CphLasso, CphElastic, CoxBoost, WeibullAFT, \
                             RSF, XGBLinear, XGBTree, XGBDart
from sklearn.model_selection import train_test_split
from utility.survival import paired_ttest_5x2cv
from tools import file_reader

N_REPEATS = 5
N_SPLITS = 5

def main():
    df = file_reader.read_csv(Path.joinpath(cfg.PROCESSED_DATA_DIR, 'home_care_ma.csv'))
    X = df.drop(['Observed', 'Weeks'], axis=1)
    y = np.array(list(tuple(x) for x in df[['Observed', 'Weeks']].to_numpy()),
                 dtype=[('Observed', 'bool'), ('Weeks', '<f8')])

    models = [Cph, CphRidge, CphLasso, CphElastic, RSF, CoxBoost, XGBLinear, XGBTree, XGBDart, WeibullAFT]
    ft_selectors = [NoneSelector, LowVar, SelectKBest10, SelectKBest20, RFE10, RFE20, RegMRMR10, RegMRMR20]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    T1, HOS = (X_train, y_train), (X_test, y_test)

    print(f"Started evaluation of {len(models)} models/{len(ft_selectors)} ft selectors/{len(T1[0])} total samples")
    for model_builder in models:
        model_name = model_builder.__name__
        model_results = pd.DataFrame()
        for ft_selector_builder in ft_selectors:
            ft_selector_name = ft_selector_builder.__name__
            for n_repeat in range(N_REPEATS):
                kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=n_repeat)
                for train, test in kf.split(T1[0], T1[1]):
                    # Split data
                    ti = (T1[0].iloc[train], T1[1][train])
                    cvi = (T1[0].iloc[test], T1[1][test])

                    # Get current model and ft selector
                    if ft_selector_name == "NoneSelector":
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) None"
                    else:
                        ft_selector_print_name = f"({ft_selectors.index(ft_selector_builder)+1}) {ft_selector_name}"
                    model_print_name = f"({models.index(model_builder)+1}) {model_name}"

                    # Create model instance and find best features
                    model = model_builder().get_estimator()
                    model_class_name = model.__class__.__name__
                    if ft_selector_name in ["RegMRMR10", "RegMRMR20"]:
                        y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                        ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model)
                    elif (model_name == 'WeibullAFT' and ft_selector_name in ["NoneSelector", "LowVar", "RFE10", "RFE20"]):
                        # No support for WeibullAFT and some selectors, so skip runs
                        t, p = np.nan, np.nan
                        res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, t, p],
                                            index=["ModelName", "FtSelectorName", "NRepeat", "T", "P"])
                        model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)
                        continue
                    elif model_name == "WeibullAFT" and ft_selector_name in ["RegMRMR10", "RegMRMR20"]:
                        y_ti_mrmr = np.array([x[0] for x in ti[1]], float)
                        ft_selector = ft_selector_builder(ti[0], y_ti_mrmr, estimator=model.lifelines_model)
                    elif model_class_name == "XGBRegressor" and ft_selector_name in ["RFE10", "RFE20"]:
                        y_ti_xgb = [x[1] if x[0] else -x[1] for x in ti[1]]
                        ft_selector = ft_selector_builder(ti[0], y_ti_xgb, estimator=model)
                    else:
                        ft_selector = ft_selector_builder(ti[0], ti[1], estimator=model)
                    current_model_features = ft_selector.get_features()

                    # Make best model
                    best_model_params = regressors.XGBTree().get_best_params()
                    best_model = regressors.XGBTree().make_model(best_model_params)
                    best_model_features = LowVar(cvi[0], cvi[1], best_model).get_features()

                    # Use best hyperparams from previous run
                    best_params = model_builder().get_best_params()
                    current_model = model_builder().get_estimator(best_params)

                    # Find stats difference between best model and current model via CV
                    t, p = paired_ttest_5x2cv(estimator1=best_model,
                                              estimator2=current_model,
                                              estimator1_best_fts=best_model_features,
                                              estimator2_best_fts=current_model_features,
                                              X=cvi[0], y=cvi[1], random_seed=0)
                    print(f"Evaluated {model_print_name} - {ft_selector_print_name} - {round(t, 3)} - {round(p, 3)}")

                    # Save results
                    res_sr = pd.Series([model_print_name, ft_selector_print_name, n_repeat, t, p],
                                        index=["ModelName", "FtSelectorName", "NRepeat", "T", "P"])
                    model_results = pd.concat([model_results, res_sr.to_frame().T], ignore_index=True)

        file_writer.write_csv(Path.joinpath(cfg.REPORTS_DIR, f"{model_name}_alarm_cv_results_ttest.csv"), model_results)

if __name__ == "__main__":
    main()