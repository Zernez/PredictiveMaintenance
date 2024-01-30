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
                tte_surv = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['MedianSurvTime']
                tte_ed = results.loc[(results['Condition'] == cond) & (results['CensoringLevel'] == cens) & (results['ModelName'] == model_name)]['EDTarget']
                median_tte_surv = round(np.median(tte_surv.dropna()), 1)
                median_tte_ed = round(np.median(tte_ed.dropna()), 1)
                error = round(median_tte_surv-median_tte_ed, 2)
                text += f"{tte_surv} & {tte_ed} & {error} "
                if cens == 0.75:
                   text += "\\\\"
                else:
                   text += " & "
            print(text)
        print()