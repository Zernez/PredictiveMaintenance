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
from utility.data import get_window_size, get_lag, get_lmd
from utility.event import EventManager

new_dataset = False
dataset = "xjtu"
axis = "X"
n_boot = 0
dataset_path = cfg.DATASET_PATH_XJTU
n_bearing = cfg.N_REAL_BEARING_XJTU
bearing_ids = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    data_util = DataETL(dataset, n_boot)
    event_manager = EventManager(dataset)
    
    if new_dataset == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in [0, 1, 2]:
        timeseries_data, frequency_data = FileReader(dataset, dataset_path).read_data(test_condition, axis=axis)
        event_times = EventManager(dataset).get_event_times(frequency_data, test_condition, lmd=get_lmd(test_condition))
        data = pd.DataFrame()
        for bearing_id in bearing_ids:
            event_time = event_times[bearing_id-1]
            transformed_data = data_util.make_moving_average(timeseries_data, event_time, bearing_id,
                                                            get_window_size(test_condition),
                                                            get_lag(test_condition))
            data = pd.concat([data, transformed_data], axis=0)
        data = data.drop(cfg.FREQUENCY_FTS + cfg.NOISE_FT, axis=1)
        data = data.reset_index(drop=True)
        for pct in cfg.CENSORING_LEVELS:
            data_censored = Formatter.add_random_censoring(data.copy(), pct=pct)
            fig, ax = plt.subplots()
            val, bins, patches = plt.hist((data_censored["Survival_time"][data_censored["Event"]],
                                           data_censored["Survival_time"][~data_censored["Event"]]),
                                          bins=10,
                                          stacked=False)
            plt.legend(patches, ["Event time (" + r"$\delta_{i} = 1$" + ")", "Censoring time (" + r"$\delta_{i} = 0$" + ")"])
            plt.xlabel("Time (min)")
            plt.ylabel("Number of occurrences")
            plt.tight_layout()
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)
            plt.savefig(f'{cfg.PLOTS_PATH}/event_times_C{test_condition+1}_cens_{int(pct*100)}.pdf',
                        format='pdf', bbox_inches="tight")
        