import numpy as np 
import pandas as pd
import re
import scipy.stats as st
import statsmodels.stats.weightstats as stat 
from scipy.stats import entropy
import config as cfg

class Event:
        
    def __init__ (self, dataset, bootstrap):
        if dataset == "xjtu":
            self.real_bearings = cfg.N_REAL_BEARING_XJTU
            self.total_bearings = self.real_bearings * (2 + bootstrap) * 2
            self.dataset_condition = cfg.RAW_DATA_PATH_XJTU
            self.base_dynamic_load = cfg.BASE_DYNAMIC_LOAD_XJTU
        elif dataset == "pronostia":
            self.real_bearings = cfg.N_BEARING_TOT_PRONOSTIA
            self.total_bearings = self.real_bearings * (2 + bootstrap) * 2  
            self.dataset_condition = cfg.RAW_DATA_PATH_PRONOSTIA
            self.base_dynamic_load = cfg.BASE_DYNAMIC_LOAD_PRONOSTIA
         # Number of frequency bins analyzed to build the event detector
        self.frequency_bins = 5           
        self.data_points_per_window = 10
        # Percentage of error allowed added to the threshold
        self.percentage_error = 10
        # Derivative safety value that is broken when a high excursion of entropy occurs from the previous state
        self.anomaly_excursion_offset_kl = 0
        # Derivative safety value that is broken when a high excursion of SD occurs from the previous state
        self.anomaly_excursion_offset_sd = 0
        # The last break out time of KL divergence used in the SD evaluation
        self.break_out_KL = 0.0
        # Number of information for each frequency bin, each metrics and bearing: 0: threshold, 1: threshold + error, 2: breakpoint, 3: length of the dataset (windows)
        self.data_information = 3 + 1

    def evaluator_KL (self, 
            x: pd.DataFrame, 
            data_points_per_window: int
        ) -> np.ndarray:

        """
        Calculate the Kullback-Leibler (KL) divergence between the reference window and the moving window for each bearing.

        Parameters:
        - x (DataFrame): DataFrame containing the data points for each bearing.
        - data_points_per_window (int): Number of data points in each window.

        Returns:
        - results (numpy.ndarray): Matrix containing the KL divergence values for each bearing and window.
        """

        # Calculate the number of windows for each bearing and initialize the matrix for the results
        length = int((len(x) - x.iloc[:, 0:1].isna().sum().values[0]) / data_points_per_window)
        len_col = len(x.columns)
        results = np.zeros((len_col, length - 1), dtype=float) 

        # For each bearing, calculate the entropy between the reference window and the moving window
        for BIN, BIN_NAME in enumerate(x.columns):
            # For each window, calculate the entropy between the reference window and the moving window
            for WINDOW, index_actual in enumerate(range(data_points_per_window, len(x), data_points_per_window)):
                index_future = index_actual + data_points_per_window

                # Check for the end of dataset and save the moving window eventually
                if index_future <= len(x) and not x[index_actual:index_future].isnull().values.any():
                    temp_window_reference = x[0:data_points_per_window]
                else:
                    break

                # Calculate the entropy between the reference window and the moving window
                moving_window = x[index_actual:index_future]
                results[BIN][WINDOW] = entropy(temp_window_reference[BIN_NAME].values, moving_window[BIN_NAME].values)

        return results

    def evaluator_SD (self, 
            x: pd.DataFrame, 
            data_points_per_window: int
        ) -> np.ndarray:

        """
        Calculate the Standard Deviation (SD) divergence between the reference window and the moving window for each bearing.

        Parameters:
        - x (DataFrame): DataFrame containing the data points for each bearing.
        - data_points_per_window (int): Number of data points in each window.

        Returns:
        - results (numpy.ndarray): Matrix containing the SD values for each bearing and window.
        """
        
        # Calculate the number of windows for each bearing and initialize the matrix for the results
        length = int((len(x) - x.iloc[:, 0:1].isna().sum().values[0]) / data_points_per_window)
        len_col = len(x.columns)
        results = np.zeros((len_col, length - 1), dtype=float) 

        # For each bearing, calculate the entropy between the reference window and the moving window
        for BIN, BIN_NAME in enumerate(x.columns):
            # For each window, calculate the entropy between the reference window and the moving window
            for WINDOW, index_actual in enumerate(range(data_points_per_window, len(x), data_points_per_window)):
                index_future = index_actual + data_points_per_window

                # Check for the end of dataset and save the moving window eventually
                if index_future <= len(x) and not x[index_actual:index_future].isnull().values.any():
                    temp_window_reference = x[index_actual:index_future]
                else:
                    break

                # Calculate the SD of the moving window
                results[BIN][WINDOW] = np.std(temp_window_reference[BIN_NAME].values)

        return results
    
    def evaluator_breakpoint_KL (self, 
            kl: np.ndarray,
            test_condition: int
        ) -> np.ndarray:

        """
        Calculates the thresholds and breakpoints for the KL divergence event detector.

        Args:
        - kl (list): List of KL divergence values for each frequency bin.
        - test_condition (int): The cardinal number of the test condition starting from 0.

        Returns:
        - thresholds_kl (numpy.ndarray): Matrix containing the calculated thresholds and breakpoints for each frequency bin and metric.
        """

        # Initialize the threshold and breakpoint matrix
        thresholds_kl = np.zeros((self.frequency_bins, self.data_information), dtype= float)

        #Percentage of error allowed added to the threshold
        percentage_error = self.percentage_error

        # Calculate the L10 RUL in hours given the test condition and the CDF for each window
        hz_speed, kn_load = self.datasheet_loader(test_condition)
        L10H_windowed = int(self.calculate_L10_minute(hz_speed, kn_load) / self.data_points_per_window)

        for BIN, bin in enumerate(kl):
            WINDOW_NO = 0
            # Slope of the line used to find the breakpoint
            m = 0
            # Intercept of the line used to find the breakpoint
            q = 0
            # Temporary reference for estabilish the breakline (updatable)
            fixed_y = 0

            thresholds_kl [BIN][3] = len(bin)

            if L10H_windowed >= len(bin):
                cdf_values = self.calculate_CDF(L10H_windowed)
            else:
                cdf_values = self.calculate_CDF(len(bin))

            for THRESHOLD_WINDOW, window_data in enumerate(bin):
                # Set the first value of the bin as the reference for the threshold
                if THRESHOLD_WINDOW == 0:
                    fixed_y= bin[0]
                    continue

                # Derivative safety value that is broken when a high excursion of entropy occurs
                anomaly_excursion_break_out_kl = self.anomaly_excursion_offset_kl

                # Set the break in offset given from CDF function over L10H
                break_in_offset_kl = - (1 - cdf_values[THRESHOLD_WINDOW])

                anomaly_excursion_break_out_kl += abs(break_in_offset_kl) * 4
                
                # Calculate the derivative of the line between the temporary threshold reference of the bin and the current value
                x = [0, 1]
                y = [fixed_y, bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative= coefficients[0]/coefficients[1]

                y = [bin[THRESHOLD_WINDOW -1], bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                previous_derivative = coefficients[0]/coefficients[1]

                # If the derivative is lower than the threshold or an anomal derivative occur, save the information and calculate the error 
                if derivative < break_in_offset_kl or previous_derivative > anomaly_excursion_break_out_kl:

                    if bin[THRESHOLD_WINDOW]> thresholds_kl [BIN][0]:
                        thresholds_kl [BIN][0] = fixed_y
                        thresholds_kl [BIN][1] = fixed_y + (fixed_y/100 * percentage_error)
                    
                    if previous_derivative > anomaly_excursion_break_out_kl: 
                        thresholds_kl [BIN][2] = THRESHOLD_WINDOW
                        self.break_out_KL = thresholds_kl [BIN][2]
                        break

                    # If is not at the end of the dataset
                    if thresholds_kl [BIN][3] > THRESHOLD_WINDOW + 1:                    
                        # Set up the window number for find the breakpoint in the future
                        WINDOW_NO = THRESHOLD_WINDOW + 1
                        
                        # If the derivative is lower than the threshold, the breakpoint is found by looking at the next values
                        for FUTURE in bin[THRESHOLD_WINDOW + 1:]:
                            if FUTURE > thresholds_kl [BIN][1]:
                                m = bin[WINDOW_NO] - bin[WINDOW_NO - 1]
                                q = bin[WINDOW_NO - 1]
                                thresholds_kl [BIN][2] = (thresholds_kl [BIN][1]/m) - (q/m) + (WINDOW_NO - 1)
                                break
                            WINDOW_NO += 1

                        # Save the time when the threshold is broken for the evaluation of the breakpoint by SD
                        self.break_out_KL = thresholds_kl [BIN][2]
                    break
                else:
                    # If the acutal value is higher than the threshold registred, update the threshold
                    if fixed_y < bin[THRESHOLD_WINDOW]:
                        fixed_y = bin[THRESHOLD_WINDOW]    

        return thresholds_kl

    def evaluator_breakpoint_SD (self, 
            sd: np.ndarray,
            test_condition: int
        ) -> np.ndarray:

        """
        Calculates the thresholds and breakpoints for the given frequency bins and metrics in the SD dataset.

        Args:
        - sd (list): List of frequency bins and metrics.
        - test_condition (int): The cardinal number of the test condition starting from 0.

        Returns:
        - thresholds_sd (numpy.ndarray): Matrix containing the calculated thresholds and breakpoints for each frequency bin and metric.
        """

        # Initialize the threshold and breakpoint matrix
        thresholds_sd = np.zeros((self.frequency_bins, self.data_information), dtype= float)
    
        break_out_KL = self.break_out_KL
        percentage_error = self.percentage_error

        # Calculate the L10 RUL in hours given the test condition and the CDF for each window
        hz_speed, kn_load = self.datasheet_loader(test_condition)
        L10H_windowed = int(self.calculate_L10_minute(hz_speed, kn_load) / self.data_points_per_window)

        for BIN, bin in enumerate(sd):
            WINDOW_NO = 0
            # Slope of the line used to find the breakpoint
            m = 0
            # Intercept of the line used to find the breakpoint
            q = 0
            # Temporary reference for estabilish the breakline (updatable)
            fixed_y = 0

            thresholds_sd [BIN][3] = len(bin)

            if L10H_windowed >= len(bin):
                cdf_values = self.calculate_CDF(L10H_windowed)
            else:
                cdf_values = self.calculate_CDF(len(bin))

            for THRESHOLD_WINDOW, window_data in enumerate(bin):
                # Set the first value of the bin as the reference for the threshold
                if THRESHOLD_WINDOW == 0:
                    fixed_y= bin[0]
                    continue

                # Derivative safety value that is broken when a high excursion of SD occurs
                anomaly_excursion_break_out_sd = self.anomaly_excursion_offset_sd

                # Set the break in offset given from CDF function over L10H
                break_in_offset_sd = - (1 - cdf_values[THRESHOLD_WINDOW])

                anomaly_excursion_break_out_sd += abs(break_in_offset_sd) * 4       
                
                # Calculate the derivative of the line between the temporary threshold reference of the bin and the current value
                x = [0, 1]
                y = [fixed_y, bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative= coefficients[0]/coefficients[1]

                y = [bin[THRESHOLD_WINDOW -1], bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                previous_derivative = coefficients[0]/coefficients[1]

                # If the derivative is lower than the threshold and KL break-threshold occurs or an anomal derivative occur, save the information and calculate the error 
                if (derivative < break_in_offset_sd and break_out_KL < THRESHOLD_WINDOW and break_out_KL > 0.0) or previous_derivative > anomaly_excursion_break_out_sd:

                    if bin[THRESHOLD_WINDOW]> thresholds_sd [BIN][0]:
                        thresholds_sd [BIN][0] = fixed_y
                        thresholds_sd [BIN][1] = fixed_y + (fixed_y/100 * percentage_error)

                    if previous_derivative > anomaly_excursion_break_out_sd: 
                        thresholds_sd [BIN][2] = THRESHOLD_WINDOW
                        break

                    # If is not at the end of the dataset
                    if thresholds_sd [BIN][3] > THRESHOLD_WINDOW + 1:
                        # Set up the window number for find the breakpoint in the future
                        WINDOW_NO = THRESHOLD_WINDOW + 1
                        
                        # If the derivative is lower than the threshold, the breakpoint is found by looking at the next values
                        for FUTURE in bin[THRESHOLD_WINDOW + 1:]:
                            if FUTURE > thresholds_sd [BIN][1]:
                                m = bin[WINDOW_NO] - bin[WINDOW_NO - 1]
                                q = bin[WINDOW_NO - 1]
                                thresholds_sd [BIN][2] = (thresholds_sd [BIN][1]/m) - (q/m) + (WINDOW_NO -1)
                                break
                            WINDOW_NO += 1          
                    break
                else:
                    #Update the derivative for the next iteration
                    previous_derivative = derivative   

        return thresholds_sd

    def make_events (self, 
            set_analytic: pd.DataFrame,
            test_condition: int
        ) -> (list, list):

        """
        Generates events for each bearing based on the given set of analytic data.

        Args:
        - set_analytic (DataFrame): The analytic data for all bearings.
        - test_condition (int): The cardinal number of the test condition starting from 0.

        Returns:
        tuple: A tuple containing two lists - events_KL and events_SD.
            - events_KL (list): The calculated thresholds and breakpoints for each bearing based on KL evaluation.
            - events_SD (list): The calculated thresholds and breakpoints for each bearing based on SD evaluation.
        """

        # Initialize the information
        evals_KL = []
        evals_SD = []
        events_KL = []
        events_SD = []      

        # For each bearing, evaluate the KL and SD
        for bearing_no in range(1, self.total_bearings + 1, 1):
            information_matrix = set_analytic[["B{}_FoH".format(bearing_no), "B{}_FiH".format(bearing_no), "B{}_FrH".format(bearing_no), "B{}_FrpH".format(bearing_no), "B{}_FcaH".format(bearing_no)]]   
            evals_KL.append(self.evaluator_KL(information_matrix, self.data_points_per_window))
            evals_SD.append(self.evaluator_SD(information_matrix, self.data_points_per_window))

        # For each bearing, calculate the threshold and breakpoint
        for eval_KL, eval_SD in zip(evals_KL, evals_SD):
            bearing_th_and_br_KL = self.evaluator_breakpoint_KL(eval_KL, test_condition)
            bearing_th_and_br_SD = self.evaluator_breakpoint_SD(eval_SD, test_condition)
            events_KL.append(bearing_th_and_br_KL)
            events_SD.append(bearing_th_and_br_SD)

        return events_KL, events_SD

    def calculate_L10_minute (self, 
            hz_speed: float, 
            kn_load: float
        ) -> float:

        """
        Calculate the L10 minute of lifetime for a given hz_speed and kn_load.

        Args:
        - hz_speed (float): The speed of the test in Hz.
        - kn_load (float): The load applied in the test in kN.

        Returns:
        - L10M (float): The L10 minute of lifetime.
        """

        # Value from the specs valid for all XJTU-SY dataset
        base_dynamic_load_C = self.base_dynamic_load * 1000
        # kN From the specs of XJTU-SY dataset of this type of testset
        dynamic_equivalent_load_P = kn_load * 1000
        # Rolling bearing coefficient 
        bearing_type_p = 3

        # Hz from the specs of XJTU-SY dataset of this type of testset
        RPM = hz_speed * 60 

        # Rotation for lifetime
        L10 = pow((base_dynamic_load_C / dynamic_equivalent_load_P), bearing_type_p)
        # Minute of lifetime 
        L10M = 1000000 * L10 * 60 / (60 * RPM)

        return L10M

    def calculate_CDF (self, 
            L10H_windowed: int
        ) -> np.ndarray:

        """
        Calculate the Cumulative Distribution Function (CDF) for a given L10H_windowed value.

        Parameters:
        - L10H_windowed (int): The L10H_windowed value for which to calculate the CDF.

        Returns:
        - cdf_values (numpy.ndarray): An array of CDF values corresponding to the x_values.
        """

        # Create an array of values for which you want to calculate the CDF
        x_values = np.linspace(0, L10H_windowed, L10H_windowed)

        # Calculate the CDF for each value in the array using numerical integration
        cdf_values = np.cumsum(self.exponential_degradation_function(x_values, L10H_windowed)) * (x_values[1] - x_values[0])

        # Normalize the CDF to be between 0 and 1
        cdf_values /= cdf_values[-1]

        return cdf_values

    def exponential_degradation_function (self, 
            x: np.ndarray, 
            L10M_windowed: int
        ) -> np.ndarray:
        
        # The decay is proportional to the L10
        decay = 1 / L10M_windowed * 2
        return L10M_windowed * np.exp(-decay * x)

    def datasheet_loader (self, 
            condition: int
        ) -> (float, float):
        
        # Given the path of the testset, extract the speed and load characteristics
        input_string = self.dataset_condition[condition]
        pattern = r'(\d+\.\d+|\d+)\D*(\d+\.\d+|\d+)'
        matches = re.search(pattern, input_string)
        hz_speed = float(matches.group(1))
        kn_load = float(matches.group(2))

        return hz_speed, kn_load


