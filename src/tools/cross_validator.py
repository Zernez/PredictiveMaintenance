import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from utility.printer import Suppressor
from utility.survival import make_event_times, convert_to_structured
from pycox.evaluation import EvalSurv

def run_cross_validation(model_builder, data, param_list, n_internal_splits):
    sample_results = pd.DataFrame()
    model_name = model_builder.__name__
    for sample in param_list:
        param_results = pd.DataFrame()
        kf = KFold(n_splits=n_internal_splits, random_state=0, shuffle=True)
        for split_idx, (train_in, test_in) in enumerate(kf.split(data[0], data[1])):
            t_train = np.array(data[1].iloc[train_in]["time"])
            e_train = np.array(data[1].iloc[train_in]["event"])
            t_test = np.array(data[1].iloc[test_in]["time"])
            e_test = np.array(data[1].iloc[test_in]["event"])
            x_train =  np.array(data[0].iloc[train_in])
            x_test =  np.array(data[0].iloc[test_in])
            times = make_event_times(t_train, e_train).astype(int)
            model = model_builder().make_model(sample)
            with Suppressor():
                if model_name in ["DeepSurv", "DSM"]:
                    model = model.fit(x_train, t_train, e_train, vsize=0.3)
                    preds = pd.DataFrame(model.predict_survival(x_test, t=list(times)), columns=times)
                elif model_name == "BNNmcd":
                    model.fit(x_train, t_train, e_train)
                    preds = pd.DataFrame(np.mean(model.predict_survival(x_test, times), axis=0), columns=times)
                else:
                    y_train = convert_to_structured(t_train, e_train)
                    model.fit(x_train, y_train)
                    preds = pd.DataFrame(np.row_stack([fn(times)
                                                       for fn in model.predict_survival_function(x_test)]), columns=times)
            pycox_eval = EvalSurv(preds.T, t_test, e_test, censor_surv="km")
            ctd = pycox_eval.concordance_td()
            res_sr = pd.Series([str(model_name), split_idx, sample, ctd],
                            index=["ModelName", "SplitIdx", "Params", "CTD"])
            param_results = pd.concat([param_results, res_sr.to_frame().T], ignore_index=True)
        mean_ctd = param_results['CTD'].mean()
        res_sr = pd.Series([str(model_name), param_results.iloc[0]["Params"], mean_ctd],
                        index=["ModelName", "Params", "CTD"])
        sample_results = pd.concat([sample_results, res_sr.to_frame().T], ignore_index=True)
    best_params = sample_results.loc[sample_results['CTD'].astype(float).idxmax()]['Params']
    return best_params