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
    model_citations = ["\cite{cox_regression_1972}", "\cite{ishwaran_random_2008}", "\cite{katzman_deepsurv_2018}", "\cite{nagpal_deep_2021}", "\cite{lillelund_uncertainty_2023}"]
    for cond in conditions:
        for index, (model_name, model_citation) in enumerate(zip(model_names, model_citations)):
            text = ""            
            text += f"& {model_name} {model_citation} & "            
            for cens in censoring:
                ci = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['CIndex']
                ibs = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['BrierScore']
                mae = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MAEHinge']
                mean_ci = round(np.mean(ci.dropna()), 2)
                mean_ibs = round(np.mean(ibs.dropna()), 2)
                mean_mae = round(np.mean(mae.dropna()), 2)
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
        