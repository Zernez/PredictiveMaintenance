import pandas as pd
import config as cfg
from glob import glob
import numpy as np
import re

def get_citation(model_name):
    if model_name == "CoxPH":
        return "\cite{cox_regression_1972}"
    elif model_name == "RSF":
        return "\cite{ishwaran_random_2008}"
    elif model_name == "DeepSurv":
        return "\cite{katzman_deepsurv_2018}"
    elif model_name == "DSM":
        return "\cite{nagpal_deep_2021}"
    return "\cite{lillelund_uncertainty_2023}"

if __name__ == "__main__":
    path = cfg.RESULTS_PATH
    all_files = glob(f'{path}/**/*.csv', recursive=True)
    conditions = [0, 1, 2]
    upsampling_methods = ['Bootstrap', 'Not_correlated', 'Correlated']
    censoring = ["10", "20", "30"]
    
    li = []
    
    upsampling_pattern = r'./results\\([^\\]+)\\'
    censoring_pattern = r'_(\d+)_results\.csv$'
    condition_pattern = r'_([^_]+)'

    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)
        
        match = re.search(upsampling_pattern, filename)
        df['Upsampling'] = match.group(1).capitalize()
        
        match = re.search(censoring_pattern, filename)
        df['Censoring'] = match.group(1)
        
        match = re.search(condition_pattern, filename)
        cond = match.group(1)
        if cond == '35Hz12kN':
            df['Condition'] = 0
        elif cond == '37.5Hz11kN':
            df['Condition'] = 1
        else:
            df['Condition'] = 2
        li.append(df)

    results = pd.concat(li, axis=0, ignore_index=True)
    
    model_names = ["CoxPH", "RSF", "DeepSurv", "DSM", "BNNmcd"]
    metrics = ["CIndex", "BrierScore", "MAEHinge"]
    
    for um in upsampling_methods:
        for cens in censoring:
            for index, model_name in enumerate(model_names):
                text = ""
                text += f"& {model_name} {get_citation(model_name)} & "
                for cond in conditions:
                    res = results.loc[(results['ModelName'] == model_name) &
                                    (results['Upsampling'] == um) &
                                    (results['Censoring'] == cens) &
                                    (results['Condition'] == cond)]
                    mean_ctd = round(np.mean(res['CIndex']), 3)
                    mean_ibs = round(np.mean(res['BrierScore']), 3)
                    mean_mae = round(np.mean(res['MAEHinge']), 3)
                    std_ctd = round(np.std(res['CIndex']), 3)
                    std_ibs = round(np.std(res['BrierScore']), 3)
                    std_mae = round(np.std(res['MAEHinge']), 3)
                    text += f"{mean_ctd}$\pm${std_ctd} & {mean_ibs}$\pm${std_ibs} & {mean_mae}$\pm${std_mae}"
                    if cond == 2:
                        text += "\\\\"
                    else:
                        text += "&"
                print(text)
    print()
