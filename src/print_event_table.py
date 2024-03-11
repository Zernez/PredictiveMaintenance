import pandas as pd
import config as cfg
from glob import glob
import numpy as np
import re

if __name__ == "__main__":
    bearing_indicies = []
    real_lifetimes = cfg.DATASHEET_LIFETIMES
    lifetimes = []
    for idx in bearing_indicies:
        lifetimes.append(real_lifetimes[f'{DATASET}_{cond_name.lower()}_b{idx}'])
    #TODO
    
    
    
    path = cfg.RESULTS_PATH
    results = pd.read_csv(f'{path}/model_results.csv', index_col=0)
    conditions = ["C1", "C2", "C3"]
    censoring = [0.25, 0.5, 0.75]
    model_names = ["CoxPH", "RSF", "DeepSurv", "DSM", "BNNmcd"]
    for cond in conditions:
        for index, model_name in enumerate(model_names):
            text = ""            
            text += f"& {model_name} & "
            for cens in censoring:
                tte_surv = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MedianSurvTime']
                tte_ed = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['EDTarget']
                median_tte_surv = round(np.median(tte_surv.dropna()), 1)
                median_tte_ed = round(np.median(tte_ed.dropna()), 1)
                error = round(median_tte_surv-median_tte_ed, 2)
                text += f"{median_tte_surv} & {median_tte_ed} & {error} "
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += "& "
            print(text)
        print()