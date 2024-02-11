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
                ci = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['CIndex']
                ibs = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['BrierScore']
                mae = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEHinge']
                mean_ci = round(np.median(ci.dropna()), 2)
                mean_ibs = round(np.median(ibs.dropna()), 2)
                mean_mae = round(np.median(mae.dropna()), 2)
                std_ci = round(np.std(ci.dropna()), 2)
                std_ibs = round(np.std(ci.dropna()), 2)
                std_mae = round(np.std(ci.dropna()), 1)
                text += f"{mean_ci}$\pm${std_ci} & {mean_ibs}$\pm${std_ibs} & {mean_mae}$\pm${std_mae}"
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += " & "
            print(text)
        print()
        