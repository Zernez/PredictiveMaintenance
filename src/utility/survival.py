import numpy as np
import pandas as pd
import math
import torch
from typing import Optional

def convert_to_structured(T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def make_event_times(t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy() 

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

class Survival:

    def __init__(self):
        pass
    
    def pred_sanitize_surv_function(self, model, X_test, times, y_test):
        surv_probs = model.predict_survival(X_test, event_times= times)
        sanitized_probs = []
        exclusor = []
        y_test_sanitized = y_test

        for counter, surv_prob in enumerate(surv_probs):
            second_layer_probs = []
            for num, singleton in enumerate(surv_prob):
                if singleton [0] < 0.5 and num not in exclusor:
                    exclusor.append(num)

        for counter, surv_prob in enumerate(surv_probs):
            second_layer_probs = []
            for num, singleton in enumerate(surv_prob):
                if num not in exclusor:
                    second_layer_probs.append(singleton)
            sanitized_probs.append(second_layer_probs)

        for idx in exclusor:
            y_test_sanitized = np.delete(y_test_sanitized, idx)

        return sanitized_probs, y_test_sanitized    
    
    def sanitize_survival_data(self, surv_preds, cvi, upper, fix_ending=False):
        # Fix ending of surv function
        if fix_ending:
            surv_preds.replace(np.nan, 1e-1000, inplace=True)
            surv_preds[math.ceil(upper)] = 1e-1000
            surv_preds.reset_index(drop=True, inplace=True)
        
        # Replace infs with 0
        surv_preds[~np.isfinite(surv_preds)] = 0

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
