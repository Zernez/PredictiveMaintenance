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

def get_upsampling_method(um):
    if um == "Bootstrap":
        return r"Bootstrap \eqref{alg:BSA}"
    elif um == "Not_correlated":
        return r"MA \eqref{alg:MAUSA}"
    return r"AMA \eqref{alg:ACMVUSA}"

if __name__ == "__main__":
    path = cfg.RESULTS_PATH
    all_files = glob(f'{path}/**/*.csv', recursive=True)
    conditions = [0, 1, 2]
    upsampling_methods = ['Bootstrap', 'Not_correlated', 'Correlated']
    censoring = ["10"]
    
    li = []
    
    upsampling_pattern = r'./results\\([^\\]+)\\'
    censoring_pattern = r'_(\d+)_results\.csv$'
    condition_pattern = r'_([^_]+_\d+)_'

    for filename in all_files:
        df = pd.read_csv(filename, index_col=0)
        
        match = re.search(upsampling_pattern, filename)
        df['Upsampling'] = re.search(upsampling_pattern, filename).group(1).capitalize()
        df['Censoring'] = re.search(censoring_pattern, filename).group(1)
        cond = re.search(condition_pattern, filename).group(1).split("_", 1)[0] # hack
        if cond == '35Hz12kN':
            df['Condition'] = 0
        elif cond == '37.5Hz11kN':
            df['Condition'] = 1
        else:
            df['Condition'] = 2
            
        li.append(df)

    results = pd.concat(li, axis=0, ignore_index=True)
    
    model_names = ["CoxPH", "RSF", "DeepSurv", "DSM", "BNNmcd"]
    metrics = ["MedianSurvTime", "EDTarget"]
    
    for um in upsampling_methods:
        for cens in censoring:
            print(r"\multirow{5}{*}{\shortstack{" + f"{get_upsampling_method(um)}" + r"\\" + f"{cens}" + r"\%}}")
            for index, model_name in enumerate(model_names):
                text = ""
                text += f"& {model_name} {get_citation(model_name)} & "
                for cond in conditions:
                    res = results.loc[(results['ModelName'] == model_name) &
                                    (results['Upsampling'] == um) &
                                    (results['Censoring'] == cens) &
                                    (results['Condition'] == cond)]
                    tte_surv = round(np.median(res["MedianSurvTime"]), 1)
                    tte_ed = round(np.median(res["EDTarget"]), 1)
                    delta = round(tte_surv-tte_ed, 2)
                    text += f"{tte_surv} & {tte_ed} & {delta} "
                    text = text.replace("nan", "NA")
                    if cond == 2:
                        text += "\\\\"
                    else:
                        text += "& "
                print(text)
            if um == "Correlated":
                break
            print(r"\cmidrule(lr){1-1}")

    print()
