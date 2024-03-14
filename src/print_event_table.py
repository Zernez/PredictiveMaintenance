import pandas as pd
import config as cfg
import numpy as np
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager
from utility.data import get_lmd

NEW_DATASET = False

if __name__ == "__main__":
    dataset = "xjtu"
    n_boot = 0
    dataset_path = cfg.DATASET_PATH_XJTU
    n_condition = len(cfg.RAW_DATA_PATH_XJTU)
    bearing_ids = list(range(1, (cfg.N_REAL_BEARING_XJTU*2)+1))
    
    if NEW_DATASET == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
    
    for test_condition in [0, 1, 2]:
        pct_error_list = list()
        _, analytic = FileReader(dataset, dataset_path).read_data(test_condition, n_boot)
        event_manager = EventManager(dataset)
        event_times = event_manager.get_event_times(analytic, test_condition,
                                                    get_lmd(test_condition))
        failure_times = event_manager.get_eol_times(analytic)
        for bearing_id in bearing_ids:
            event_time = event_times[bearing_id-1]
            failure_time = failure_times[bearing_id-1]
            error = event_time - failure_time
            pct_error = ((event_time - failure_time)/ failure_time) * 100
            pct_error_list.append(pct_error)
            print(f"{event_time} & {failure_time} & {error} & {abs(round(pct_error, 1))} \\\\")
        print()