import numpy as np
import pandas as pd
import math

class Survival:

    def __init__(self):
        pass    
    
    def sanitize_survival_data(self, surv_preds, cvi, upper):
        # Remove NaN's
        #surv_preds.replace(np.nan, 1e-1000, inplace=True)
        #surv_preds[math.ceil(upper)] = 1e-1000
        #surv_preds.reset_index(drop=True, inplace=True)
        #surv_preds[~np.isfinite(surv_preds)] = 0

        # Remove rows where first pred is <0.5
        bad_idx = surv_preds[surv_preds.iloc[:,0] < 0.5].index
        sanitized_surv_preds = surv_preds.drop(bad_idx).reset_index(drop=True)
        sanitized_cvi = np.delete(cvi, bad_idx)
        
        return sanitized_surv_preds, sanitized_cvi

    def predict_survival_function(self, model, X_test, times):
        # lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        # times = np.arange(np.ceil(lower + 1), np.floor(upper - 1), dtype=int)
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_survival_function(X_test).T
            return surv_prob
        elif model.__class__.__name__ == 'DeepCoxPH' or model.__class__.__name__ == 'DeepSurvivalMachines':
            surv_prob = pd.DataFrame(model.predict_survival(X_test, t= list(times)), columns=times)
            return surv_prob
        elif model.__class__.__name__ == 'MCD':
            surv_prob = pd.DataFrame(np.mean(model.predict_survival(X_test, event_times= times), axis=0))
            return surv_prob            
        else:
            surv_prob = pd.DataFrame(np.row_stack([fn(times) for fn in model.predict_survival_function(X_test)]), columns=times)
            return surv_prob

    def predict_hazard_function(self, model, X_test, times):
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_cumulative_hazard(X_test)
            return surv_prob
        elif model.__class__.__name__ == 'DeepCoxPH' or model.__class__.__name__ == 'DeepSurvivalMachines':
            risk_pred = model.predict_risk(X_test, t= times).flatten()
            return risk_pred
        elif model.__class__.__name__ == 'MCD':
            risk_pred = np.mean(model.predict_risk(X_test, event_times= times), axis= 0).flatten()
            return risk_pred             
        else:
            surv_prob = np.row_stack([fn(times) for fn in model.predict_cumulative_hazard_function(X_test)])
            return pd.DataFrame(surv_prob, columns=times)
