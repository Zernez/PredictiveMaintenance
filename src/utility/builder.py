import os
import pandas as pd
import re
from tools.featuring import Featuring
import config as cfg


class Builder:

    def __init__ (self, dataset, bootstrap):
        if dataset == "xjtu":
            self.real_bearing = cfg.N_REAL_BEARING_XJTU
            boot_folder_size = (2 + bootstrap) * 2
            self.total_bearings= self.real_bearing * boot_folder_size
            self.raw_main_path= cfg.RAW_DATA_PATH_XJTU
            self.aggregate_main_path= cfg.DATASET_PATH_XJTU

        elif dataset == "pronostia":
            self.real_bearing = cfg.N_REAL_BEARING_PRONOSTIA
            boot_folder_size = (2 + bootstrap) * 2
            self.total_bearings= self.real_bearing * boot_folder_size
            self.raw_main_path= cfg.RAW_DATA_PATH_PRONOSTIA
            self.aggregate_main_path= cfg.DATASET_PATH_PRONOSTIA
        self.dataset= dataset 

    def build_new_dataset (self, 
            bootstrap: int = 0
        ): 

        self.from_raw_to_csv(bootstrap)
        self.aggregate_and_refine()

    def from_raw_to_csv (self, 
            bootstrap: int = 0
        ):

        """
        Create from raw data of two axis vibration data into timeseries data.
        The timeseries is structured to have a row of feature for each file of raw data and save into CSV format.

        Args:
        - bootstrap (int): The bootstrap multiplier value.

        Returns:
        - None
        """

        # For each type of test condition start to create the timeseries data
        for TYPE_TEST, group in enumerate(self.raw_main_path):
            BEARING_CHANNEL_BOOTSTRAPPED = 1

            # For each real bearing create the timeseries data
            for bearing in range(1, self.real_bearing + 1, 1):
                dataset_path = group + "Bearing1_" + str(bearing)
                if self.dataset == "xjtu":
                    datasets, bootstrap_val = Featuring.time_features_xjtu(dataset_path, bootstrap = bootstrap)
                elif self.dataset == "pronostia":
                    datasets, bootstrap_val = Featuring.time_features_pronostia(dataset_path, bootstrap = bootstrap)

                # Create the label for each column and the filename for each dataset, hardcoded for the two axis vibration data bootstrap
                BEARING_BOOTSTRAPPED = 1
                for dataset in datasets:
                    dataset.columns = ['B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_mean', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_std', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_skew', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_kurtosis', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_entropy',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_rms', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_max', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_p2p', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_crest', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_clearence',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_shape', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_impulse', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_FoH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_FiH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_FrH',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_FrpH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_FcaH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Fo', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Fi', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Fr', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Frp', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Fca',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_noise', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Event', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED) + '_Survival_time',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_mean', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_std', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_skew', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_kurtosis', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_entropy',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_rms', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_max', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_p2p', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_crest', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_clearence',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_shape', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_impulse', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_FoH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_FiH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_FrH',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_FrpH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_FcaH', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Fo', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Fi', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Fr', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Frp', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Fca',
                                       'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_noise', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Event', 'B' + str(BEARING_CHANNEL_BOOTSTRAPPED + 1) + '_Survival_time']

                    dataname = self.aggregate_main_path + "Bearing1_" + str(bearing) + "_" + str(BEARING_BOOTSTRAPPED) + "_timefeature" + "_" + str(TYPE_TEST) + ".csv"
                    dataset.to_csv(dataname, index=False)
                    BEARING_BOOTSTRAPPED += 1
                    BEARING_CHANNEL_BOOTSTRAPPED += 2

                # Save the bootstrap information in a csv file that will be used for build and upsample the survival dataset
                dataname = self.aggregate_main_path + "Bearing1_" + str(bearing) + "_bootstrap" + "_" + str(TYPE_TEST) + ".csv"
                bootstrap_val.to_csv(dataname, index=False)

    def aggregate_and_refine (self):
        """
        Final step that aggregate and refine the data for event detection and survival analysis after being processed from prerocessed CSV files.
        It saves the aggregated data in separate CSV files for each type of data and information.

        Returns:
        - None
        """

        # Data used for the event detector
        set_analytic = pd.DataFrame()
        # Covariates used for survival analysis
        set_covariates = pd.DataFrame()
        # Bootstrap values used for upsample the survival analysis dataset
        set_boot = pd.DataFrame()

        # For each type of test condition, aggregate the data
        for TYPE_TEST, group in enumerate(self.raw_main_path):

            # For each CSV file of timeseries data, using regex condense the information into a single dataframe
            for filename in os.listdir(self.aggregate_main_path):
                if re.search('^Bearing.*timefeature_' + str(TYPE_TEST), filename):
                    
                    #From the dataframe of the csv file, select the columns of interest like time features, frequency features
                    datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename))
                    set_analytic_aux = datafile.iloc[:, 12: 17]
                    set_cov_aux = datafile.iloc[:, 0: 25]
                    set_analytic = pd.concat([set_analytic, set_analytic_aux], axis=1)
                    set_covariates = pd.concat([set_covariates, set_cov_aux], axis=1)

                    set_analytic_aux = datafile.iloc[:, 37: 42]
                    set_cov_aux = datafile.iloc[:, 25: 52]
                    set_analytic = pd.concat([set_analytic, set_analytic_aux], axis=1)
                    set_covariates = pd.concat([set_covariates, set_cov_aux], axis=1)

                elif re.search('^Bearing.*bootstrap_' + str(TYPE_TEST), filename):
                    col_label = re.findall("_\d", filename)[0][1]
                    datafile = pd.read_csv(os.path.join(self.aggregate_main_path, filename), index_col=False)
                    datafile.rename(columns={'Bootstrap values': col_label}, inplace=True)
                    set_boot = pd.concat([set_boot, datafile], axis=1)

            # Save the aggregated data in separate CSV files
            set_analytic.to_csv(self.aggregate_main_path + 'analytic_' + str(TYPE_TEST) + '.csv', index=False)
            set_covariates.to_csv(self.aggregate_main_path + 'covariates_' + str(TYPE_TEST) + '.csv', index=False)
            set_boot.to_csv(self.aggregate_main_path + 'boot_' + str(TYPE_TEST) + '.csv', index=False)

            # Clean the variables for the next batch of data
            set_analytic = pd.DataFrame()
            set_covariates = pd.DataFrame()
            set_boot = pd.DataFrame()
