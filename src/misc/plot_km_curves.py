import numpy as np
import pandas as pd
import config as cfg
from sksurv.util import Surv
from utility.builder import Builder
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
import matplotlib.pyplot as plt
from tools.formatter import Formatter
from xgbse.non_parametric import calculate_kaplan_vectorized
from utility.survival import make_event_times
from tools.data_loader import DataLoader

DATASET_NAME = "xjtu"
AXIS = "X"
BEARING_IDS = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    for condition in [0, 1, 2]:
        for pct in cfg.CENSORING_LEVELS:
            dl = DataLoader(DATASET_NAME, AXIS, condition).load_data()
            data = pd.DataFrame()
            for bearing_id in BEARING_IDS:
                df = dl.make_moving_average(bearing_id)
                df = Formatter.add_random_censoring(df, pct)
                df = df.sample(frac=1, random_state=0)
                data = pd.concat([data, df], axis=0)
            data = data.reset_index(drop=True)
            y = Surv.from_dataframe("Event", "Survival_time", data)
            continuous_times = make_event_times(np.array(y['Survival_time']), np.array(y['Event'])).astype(int)
            continuous_times = np.unique(continuous_times)
            fig = plt.figure(figsize=(6, 4))
            km_mean, km_high, km_low = calculate_kaplan_vectorized(y['Survival_time'].reshape(1,-1),
                                                                   y['Event'].reshape(1,-1),
                                                                   continuous_times)
            plt.plot(km_mean.columns, km_mean.iloc[0], 'k--', linewidth=2, alpha=1, label=r"$\mathbb{E}[S(t)]$ Kaplan-Meier", color="black")
            plt.fill_between(km_mean.columns, km_low.iloc[0], km_high.iloc[0], alpha=0.2, color="black")
            plt.xlabel("Time (min)")
            plt.ylabel("Survival probability S(t)")
            plt.tight_layout()
            plt.grid()
            plt.legend()
            plt.savefig(f'{cfg.PLOTS_PATH}/kaplan_meier_C{condition+1}_cens_{int(pct*100)}.pdf', format='pdf', bbox_inches="tight")
        