import numpy as np
import pandas as pd
from time import time
import math
import argparse
import warnings
import config as cfg
import re
from pycox.evaluation import EvalSurv
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgbse.metrics import approx_brier_score
from sklearn.model_selection import RandomizedSearchCV
from tools.feature_selectors import NoneSelector, LowVar, SelectKBest4, SelectKBest8, RegMRMR4, RegMRMR8, UMAP8, VIF4, VIF8, PHSelector
from tools.regressors import CoxPH, CphRidge, CphLASSO, CphElastic, RSF, CoxBoost, GradientBoostingDART, WeibullAFT, LogNormalAFT, LogLogisticAFT, DeepSurv, DSM
from tools.file_reader import FileReader
from tools.data_ETL import DataETL
from utility.builder import Builder
from tools.experiments import SurvivalRegressionCV
from utility.survival import Survival
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from lifelines import WeibullAFTFitter
from sklearn.model_selection import ParameterSampler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

N_BOOT = 3
PLOT = True
RESUME = True
NEW_DATASET = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=False,
                        default=None)
    parser.add_argument('--typedata', type=str,
                        required=False,
                        default=None)
    parser.add_argument('--merge', type=str,
                        required=False,
                        default=None)
    args = parser.parse_args()

    global DATASET
    global TYPE
    global MERGE
    global N_CONDITION
    global N_BEARING
    global N_SPLITS
    global TRAIN_SIZE

    if args.dataset:
        DATASET = args.dataset

    if args.typedata:
        TYPE = args.typedata
    
    if args.merge:
        MERGE = args.merge
    
    DATASET = "xjtu"
    TYPE = "correlated"
    MERGE = False

    if DATASET == "xjtu":
        N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)
        N_BEARING = cfg.N_REAL_BEARING_XJTU
        TRAIN_SIZE = 0.7
    elif DATASET == "pronostia":
        N_CONDITION = len(cfg.RAW_DATA_PATH_PRONOSTIA)
        N_BEARING = cfg.N_REAL_BEARING_PRONOSTIA
        TRAIN_SIZE = 0.5
    
    if NEW_DATASET== True:
        Builder(DATASET).build_new_dataset(bootstrap=N_BOOT)
        
    survival = Survival()
    
    cov_group = []
    boot_group = []
    info_group = []
    for i in range (0, N_CONDITION):
        cov, boot, info_pack = FileReader(DATASET).read_data(i)
        cov_group.append(cov)
        boot_group.append(boot)
        info_group.append(info_pack)

    data_container_X = []
    data_container_y= []
    if MERGE == True:
        data_X_merge = pd.DataFrame()
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_temp_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            if i== 0:
                deltaref_y_merge =  deltaref_temp_y
            else:
                deltaref_y_merge =  deltaref_y_merge.update(deltaref_temp_y)
            data_X_merge = pd.concat([data_X_merge, data_temp_X], ignore_index=True)
        data_container_X.append(data_X_merge)
        data_container_y.append(deltaref_y_merge)
    else:
        for i, (cov, boot, info_pack) in enumerate(zip(cov_group, boot_group, info_group)):
            data_temp_X, deltaref_y = DataETL(DATASET).make_surv_data_sklS(cov, boot, info_pack, N_BOOT, TYPE)
            data_container_X.append(data_temp_X)
            data_container_y.append(deltaref_y)
                                                                          
    for i, (data_X, data_y) in enumerate(zip(data_container_X, data_container_y)):

        data_train_index = [0, 1, 2]
        data_test_index = [3, 4]
                        
        data_train_X_merge = pd.DataFrame()
        for element in data_train_index:
            data_train_X_merge = pd.concat([data_train_X_merge, data_X[element]], ignore_index=True)
        
        data_test_X_merge = pd.DataFrame()
        for element in data_test_index:
            data_test_X_merge = pd.concat([data_test_X_merge, data_X[element]], ignore_index=True)

        data_train_X = data_train_X_merge
        data_test_X = data_test_X_merge
        times = np.arange(0, 123, 1)
        
        data_train_y = Surv.from_dataframe("Event", "Survival_time", data_train_X)
        data_test_y = Surv.from_dataframe("Event", "Survival_time", data_test_X)
        
        data_util = DataETL(DATASET)
        S12, S22 = (data_train_X, data_train_y), (data_test_X, data_test_y)
        set_tr2, set_te2, set_tr_NN2, set_te_NN2 = data_util.format_main_data(S12, S22)
        set_tr2, set_te2, set_tr_NN2, set_te_NN2 = data_util.centering_main_data(set_tr2, set_te2, set_tr_NN2, set_te_NN2)

        model = RSF().make_model()
        best_params = {
                    'n_estimators': 100,
                    'max_depth' : 7,
                    'min_samples_split': 2,
                    'min_samples_leaf': 4,
                    'max_features': None,
                    'random_state': 0
                }
        model.set_params(**best_params)
        model.fit(set_tr2[0], set_tr2[1])
        surv_prob = survival.predict_survival_function(model, set_te2[0], times)
        
        plt.figure(dpi=80)
        mean_surv = np.mean(surv_prob, axis=0)
        plt.step(times, mean_surv, where="post", label="RSF mean")
        plt.ylabel("Probability of survival $S(t)$")
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
