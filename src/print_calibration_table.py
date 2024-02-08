import pandas as pd
import config as cfg
from glob import glob
import numpy as np
import re

if __name__ == "__main__":
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
                d_cal = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['DCalib']
                c_cal = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['CCalib']
                sum_d_cal = int(np.nansum(d_cal))
                sum_c_cal = int(np.nansum(c_cal))
                if model_name in ["CoxPH", "RSF", "DeepSurv", "DSM"]:
                    text += f"{sum_d_cal}/5 & - "
                else:
                    text += f"{sum_d_cal}/5 & {sum_c_cal}/5 "
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += "& "
            print(text)
        print()