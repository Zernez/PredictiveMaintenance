import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import math

class Survival:

    def __init__(self):
        pass    

    def predict_survival_function(self, model, X_test, y_test, lower, upper) -> pd.DataFrame:
#        lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        times = np.arange(math.ceil(lower), math.floor(upper+1), dtype=int)
        surv_prob = np.row_stack([fn(times) for fn in model.predict_survival_function(X_test)])
        return pd.DataFrame(surv_prob, columns=times)

    def predict_hazard_function(self, model, X_test, y_test, lower, upper) -> pd.DataFrame:
#        lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        times = np.arange(math.ceil(lower), math.floor(upper+1), dtype=int)
        surv_prob = np.row_stack([fn(times) for fn in model.predict_cumulative_hazard_function(X_test)])
        return pd.DataFrame(surv_prob, columns=times)

    def predict_td_risk_scores(self, model, X_test, y_train, y_test, time_bins) -> pd.DataFrame:
        chf_funcs = model.predict_cumulative_hazard_function(X_test)
        risk_scores = np.row_stack([chf(time_bins) for chf in chf_funcs])
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, time_bins)
        return pd.DataFrame(auc), pd.Series(mean_auc)

    def paired_ttest_5x2cv(self, estimator1, estimator2, estimator1_best_fts,
                        estimator2_best_fts, X, y, random_seed=None):
        rng = np.random.RandomState(random_seed)

        variance_sum = 0.0
        first_diff = None

        def score_diff(X_1, X_2, y_1, y_2, estimator1_best_fts, estimator2_best_fts):
            if estimator1.__class__.__name__ == "XGBRegressor":  # Manually handle XGB case
                print ("y_1_2: ", y_1)
                y_1_xgb = [x[1] if x[0] else -x[1] for x in y_1]
                estimator1.fit(X_1.loc[:, estimator1_best_fts], y_1_xgb)
                preds = estimator1.predict(X_2.loc[:, estimator1_best_fts])
                if any(np.isnan(preds)): # and any(y_2['Event']== False and y_2['Event']['Survival_time'])
                    est1_score = 0
                else:
                    est1_score = concordance_index_censored(y_2['Event'], y_2['Survival_time'], preds)[0]
            elif estimator1.__class__.__name__ == "type": # Manually handle WeibullAFT case
                wf = estimator1()
                estimator1_best_fts = np.append(estimator1_best_fts, 'Event')
                X_1_wf = pd.concat([X_1.reset_index(drop=True),
                                    pd.DataFrame(y_1['Event'], columns=['Event'])], axis=1)
                y_1_wf = np.array([x[1] for x in y_1], float)
                try:
                    wf.fit(X_1_wf.loc[:, estimator1_best_fts], y_1_wf)
                    X_2_wf = pd.concat([X_2.reset_index(drop=True),
                                        pd.DataFrame(y_2['Event'], columns=['Event'])], axis=1)
                    preds = wf.predict(X_2_wf.loc[:, estimator1_best_fts])
                    if any(np.isnan(preds)):
                        est1_score = 0
                    else:
                        est1_score = concordance_index(y_2['Survival_time'], preds, y_2['Event'])
                except:
                    est1_score = 0
            else:
                estimator1.fit(X_1.loc[:, estimator1_best_fts], y_1)
                preds = estimator1.predict(X_2.loc[:, estimator1_best_fts])
                if any(np.isnan(preds)):
                    est1_score = 0
                else:
                    est1_score = concordance_index_censored(y_2['Event'], y_2['Survival_time'], preds)[0]

            if estimator2.__class__.__name__ == "XGBRegressor":
                y_1_xgb = [x[1] if x[0] else -x[1] for x in y_1]
                estimator2.fit(X_1.loc[:, estimator2_best_fts], y_1_xgb)
                preds = estimator2.predict(X_2.loc[:, estimator2_best_fts])
                if any(np.isnan(preds)):
                    est2_score = 0
                else:
                    est2_score = concordance_index_censored(y_2['Event'], y_2['Survival_time'], preds)[0]
            elif estimator2.__class__.__name__ == "type":
                wf = estimator2()
                estimator2_best_fts = np.append(estimator2_best_fts, 'Event')
                X_1_wf = pd.concat([X_1.reset_index(drop=True),
                            pd.DataFrame(y_1['Event'], columns=['Event'])], axis=1)
                y_1_wf = np.array([x[1] for x in y_1], float)
                try:
                    wf.fit(X_1_wf.loc[:, estimator2_best_fts], y_1_wf)
                    X_2_wf = pd.concat([X_2.reset_index(drop=True),
                                        pd.DataFrame(y_2['Event'], columns=['Event'])], axis=1)
                    preds = wf.predict(X_2_wf.loc[:, estimator2_best_fts])
                    if any(np.isnan(preds)):
                        est2_score = 0
                    else:
                        est2_score = concordance_index(y_2['Survival_time'], preds, y_2['Event'])
                except:
                    est2_score = 0
            else:
                estimator2.fit(X_1.loc[:, estimator2_best_fts], y_1)
                preds = estimator2.predict(X_2.loc[:, estimator2_best_fts])
                if any(np.isnan(preds)):
                    est2_score = 0
                else:
                    est2_score = concordance_index_censored(y_2['Event'], y_2['Survival_time'], preds)[0]

            score_diff = est1_score - est2_score
            return score_diff

        for i in range(3):

            randint = rng.randint(low=0, high=32767)

            print ("X: ", X)
            print ("y:", y)


            # def train_test_split (self, X, Y, test_size):

            #     for y in Y:
                    

            #     return X_1, X_2, y_1 y_2
                

            X_1, X_2, y_1, y_2 = train_test_split(X, y, 0.5)

            print ("X_1: ", X_1)
            print ("X_2: ", X_2)
            print ("y_1: ", y_1)
            print ("y_2: ", y_2)

            score_diff_1 = score_diff(X_1, X_2, y_1, y_2, estimator1_best_fts, estimator2_best_fts)
            score_diff_2 = score_diff(X_2, X_1, y_2, y_1, estimator1_best_fts, estimator2_best_fts)
            score_mean = (score_diff_1 + score_diff_2) / 2.0
            score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
            variance_sum += score_var
            if first_diff is None:
                first_diff = score_diff_1

        numerator = first_diff
        denominator = np.sqrt(1 / 5.0 * variance_sum)
        t_stat = numerator / denominator if denominator else 0

        pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
        return float(t_stat), float(pvalue)