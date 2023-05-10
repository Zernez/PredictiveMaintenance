import os
import pandas as pd
import numpy as np
import re


class Aggregator:

    def __init__(self):
        pass

    def aggregate_to_csv (self):
        dataset_path = "./data/XJTU-SY/csv/"
        set_analytic= pd.DataFrame()
        set_covariates= pd.DataFrame()
        set_boot= pd.DataFrame()

        for filename in os.listdir(dataset_path):

            i= 0

            if re.search('^Bearing.*timefeature', filename):
                datafile = pd.read_csv(os.path.join(dataset_path, filename))
                set_analytic_aux= datafile.iloc[:, 12: 17]
                set_cov_aux= datafile.iloc[:, 0: 12]
                set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                set_cov_aux= datafile.iloc[:, 17: 19]     
                set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)   

                set_analytic_aux= datafile.iloc[:, 31: 36]
                set_cov_aux= datafile.iloc[:, 19: 31]
                set_analytic= pd.concat([set_analytic, set_analytic_aux], axis= 1)
                set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)
                set_cov_aux= datafile.iloc[:, 36: 38]     
                set_covariates= pd.concat([set_covariates, set_cov_aux], axis= 1)

            elif re.search('^Bearing.*bootstrap', filename):
                col_label = re.findall("_\d", filename)[0][1]      
                datafile = pd.read_csv(os.path.join(dataset_path, filename), index_col=False)
                datafile.rename(columns = {'Bootstrap values': col_label}, inplace = True)
                set_boot = pd.concat([set_boot, datafile], axis= 1)       

        # check_nan = [set_analytic.isnull().values.any(), set_covariates.isnull().values.any()]
        # print(check_nan)

        set_analytic.to_csv(dataset_path + 'analytic.csv', index= False)
        set_covariates.to_csv(dataset_path + 'covariates.csv', index= False)
        set_boot.to_csv(dataset_path + 'boot.csv', index= False)