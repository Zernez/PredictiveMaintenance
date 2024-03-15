import pandas as pd
import config as cfg
import numpy as np
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager
from tools.data_ETL import DataETL
from utility.data import get_window_size, get_lag, get_lmd
from lifelines import CoxPHFitter
from textwrap import fill
import numpy as np
from lifelines.statistics import proportional_hazard_test, TimeTransformers
from lifelines.utils import format_p_value
from lifelines.utils.lowess import lowess

NEW_DATASET = False

pd.set_option('display.float_format', lambda x: '%.3f' % x)

if __name__ == "__main__":
    dataset = "xjtu"
    n_boot = 0
    dataset_path = cfg.DATASET_PATH_XJTU
    n_condition = len(cfg.RAW_DATA_PATH_XJTU)
    n_bearing = cfg.N_REAL_BEARING_XJTU
    bearing_ids = list(range(1, (n_bearing*2)+1))
    
    if NEW_DATASET == True:
        Builder(dataset, n_boot).build_new_dataset(bootstrap=n_boot)
        
    data_util = DataETL(dataset, n_boot)
    
    results = pd.DataFrame()
    covariate_names = ["mean", "std", "skew", "kurtosis", "entropy", "rms",
                       "max", "p2p", "crest", "clearence", "shape", "impulse"]
    results['Covariates'] = covariate_names
    for test_condition in [0, 1, 2]:
        data = pd.DataFrame()
        covariates, analytic = FileReader(dataset, dataset_path).read_data(test_condition, n_boot)
        event_manager = EventManager(dataset)
        event_times = event_manager.get_event_times(analytic, test_condition, get_lmd(test_condition))
        failure_times = event_manager.get_eol_times(analytic)
        
        for bearing_id in bearing_ids:
            event_time = event_times[bearing_id-1]
            transformed_data = data_util.make_moving_average(covariates, event_time, bearing_id,
                                                             get_window_size(test_condition),
                                                             get_lag(test_condition))
            data = pd.concat([data, transformed_data], axis=0)
            data = data.reset_index(drop=True)
        cph = CoxPHFitter()
        df = data.loc[:, ~data.columns.isin(['Fca', 'Fi', 'Fo', 'Fr', 'Frp', 'FoH', 'FiH',
                                             'FrH', 'FrpH', 'FcaH', 'noise'])]
        cph.fit(df, duration_col="Survival_time", event_col= "Event")
        residuals = cph.compute_residuals(df, kind="scaled_schoenfeld")
        test_results = proportional_hazard_test(cph, df, time_transform=["rank"],
                                                precomputed_residuals=residuals)
        results[f'C{test_condition+1}'] = test_results.p_value
    print(results)
        