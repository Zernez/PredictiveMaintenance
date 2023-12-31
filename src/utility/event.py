import numpy as np 
import pandas as pd
import scipy.stats as st
import statsmodels.stats.weightstats as stat 
from scipy.stats import entropy
import config as cfg

class Event:
        
    def __init__ (self, dataset, bootstrap):
        if dataset == "xjtu":
            self.real_bearings = cfg.N_REAL_BEARING_XJTU
            self.total_bearings = self.real_bearings * (2 + bootstrap) * 2
        elif dataset == "pronostia":
            self.real_bearings = cfg.N_BEARING_TOT_PRONOSTIA
            self.total_bearings = self.real_bearings * (2 + bootstrap) * 2  
        self.frequency_bins = 5            
        self.data_points_per_window = 10
        self.percentage_error = 10
        self.break_in_offset_init_kl = -0.8
        self.break_in_offset_init_sd = -0.6
        self.break_out_offset_init_kl = 0.6
        self.break_out_offset_init_sd = 0.4
        self.break_out_KL = 0.0

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
            kl: np.ndarray
        ) -> np.ndarray:

        """
        Calculates the thresholds and breakpoints for the KL divergence event detector.

        Args:
        - kl (list): List of KL divergence values for each frequency bin.

        Returns:
        - thresholds_kl (numpy.ndarray): Matrix containing the calculated thresholds and breakpoints for each frequency bin and metric.
        """

        # Number of frequency bins used to analyze and build the event detector
        bins = self.frequency_bins
        # Number of information (3 important info plus 1 side info) for each frequency bin, each metrics and bearing: 
        # 0: threshold, 1: threshold + error, 2: breakpoint, 3: length of the dataset
        data = 3 + 1
        #Percentage of error allowed added to the threshold
        percentage_error = self.percentage_error
        # Derivative value to break used for find the threshold in KD
        break_in_offset_init_kl = self.break_in_offset_init_kl
        # Derivative value to break used for find the threshold in KL in a successive postive increasing case
        break_out_offset_init_kl = self.break_out_offset_init_kl
        # Initialize the threshold and breakpoint matrix
        thresholds_kl = np.zeros((bins, data), dtype= float)

        for BIN, bin in enumerate(kl):
            WINDOW_NO = 0
            # Slope of the line used to find the breakpoint
            m = 0
            # Intercept of the line used to find the breakpoint
            q = 0
            # Temporary reference for estabilish the breakline (updatable)
            fixed_y = 0
            break_in_offset_kl = break_in_offset_init_kl
            break_out_offset_kl = break_out_offset_init_kl
            previous_derivative = 0
            thresholds_kl [BIN][3] = len(bin)

            for THRESHOLD_WINDOW, window_data in enumerate(bin):
                # Set the first value of the bin as the reference for the threshold
                if THRESHOLD_WINDOW == 0:
                    fixed_y= bin[0]
                    continue
                
                # Calculate the derivative of the line between the temporary threshold reference of the bin and the current value
                x = [0, 1]
                y = [fixed_y, bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative= coefficients[0]/coefficients[1]

                y = [bin[THRESHOLD_WINDOW -1], bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative_last= coefficients[0]/coefficients[1]

                # If the derivative is lower than the threshold, save the information and calculate the error 
                if derivative < break_in_offset_kl or derivative_last > 3:

                    if bin[THRESHOLD_WINDOW]> thresholds_kl [BIN][0]:
                        thresholds_kl [BIN][0] = fixed_y
                        thresholds_kl [BIN][1] = fixed_y + (fixed_y/100 * percentage_error)

                    if thresholds_kl [BIN][3] > THRESHOLD_WINDOW:                    
                        # Set up the window number for find the breakpoint in the future
                        WINDOW_NO = THRESHOLD_WINDOW + 1
                        
                        # If the derivative is lower than the threshold, the breakpoint is found by looking at the next values
                        for FUTURE in bin[THRESHOLD_WINDOW + 1:]:
                            if FUTURE > thresholds_kl [BIN][1]:
                                m = bin[WINDOW_NO]-bin[WINDOW_NO - 1]
                                q = bin[WINDOW_NO - 1]
                                thresholds_kl [BIN][2] = (thresholds_kl [BIN][1]/m) - (q/m) + (WINDOW_NO -1)
                                break
                            WINDOW_NO += 1

                            self.break_out_KL = thresholds_kl [BIN][2]
                    break
                else:
                    # If the derivative is higher than the threshold, the threshold is updated to be more sensitive next window 
                    if break_in_offset_kl + 0.15 <= 0.0: 
                        break_in_offset_kl += 0.15 
                    else:
                        break_in_offset_kl = 0.0
                    # If the acutal value is higher than the threshold registred, update the threshold
                    if fixed_y < bin[THRESHOLD_WINDOW]:
                        fixed_y = bin[THRESHOLD_WINDOW]

                    #Update the derivative for the next iteration
                    previous_derivative = derivative        

        return thresholds_kl

    def evaluator_breakpoint_SD (self, 
            sd: np.ndarray
        ) -> np.ndarray:

        """
        Calculates the thresholds and breakpoints for the given frequency bins and metrics in the SD dataset.

        Args:
        - sd (list): List of frequency bins and metrics.

        Returns:
        - thresholds_sd (numpy.ndarray): Matrix containing the calculated thresholds and breakpoints for each frequency bin and metric.
        """

        # Number of frequency bins used to analyze and build the event detector
        bins = self.frequency_bins
        # Number of information (3 important info plus 1 side info) for each frequency bin, each metrics and bearing: 
        # 0: threshold, 1: threshold + error, 2: breakpoint, 3: length of the dataset
        data = 3 + 1
        #Percentage of error allowed added to the threshold
        percentage_error = self.percentage_error
        # Derivative value to break used for find the threshold in SD
        break_in_offset_init_sd = self.break_in_offset_init_sd
        # Derivative value to break used for find the threshold in SD in a successive postive increasing case
        break_out_offset_init_sd = self.break_out_offset_init_sd
        # Initialize the threshold and breakpoint matrix
        thresholds_sd = np.zeros((bins, data), dtype=float)

        for BIN, bin in enumerate(sd):
            WINDOW_NO = 0
            # Slope of the line used to find the breakpoint
            m = 0
            # Intercept of the line used to find the breakpoint
            q = 0
            # Temporary reference for estabilish the breakline (updatable)
            fixed_y = 0
            break_in_offset_sd= break_in_offset_init_sd
            break_out_offset_sd = break_out_offset_init_sd
            previous_derivative = 0
            thresholds_sd [BIN][3] = len(bin)

            for THRESHOLD_WINDOW, window_data in enumerate(bin):
                # Set the first value of the bin as the reference for the threshold
                if THRESHOLD_WINDOW == 0:
                    fixed_y = bin[0]
                    continue

                # Calculate the derivative of the line between the temporary threshold reference of the bin and the current value
                x = [0, 1]
                y = [fixed_y, bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative= coefficients[0]/coefficients[1]
                
                y = [bin[THRESHOLD_WINDOW -1], bin[THRESHOLD_WINDOW]]
                coefficients = np.polyfit(x, y, 1)
                derivative_last= coefficients[0]/coefficients[1]

                # If the derivative is lower than the threshold, save the information and calculate the error 
                if (derivative < break_in_offset_sd and self.break_out_KL < THRESHOLD_WINDOW) or derivative_last > 3:

                    if bin[THRESHOLD_WINDOW]> thresholds_sd [BIN][0]:
                        thresholds_sd [BIN][0] = fixed_y
                        thresholds_sd [BIN][1] = fixed_y + (fixed_y/100 * percentage_error)

                    if thresholds_sd [BIN][3] > THRESHOLD_WINDOW:
                        # Set up the window number for find the breakpoint in the future
                        WINDOW_NO = THRESHOLD_WINDOW + 1
                        
                        # If the derivative is lower than the threshold, the breakpoint is found by looking at the next values
                        for FUTURE in bin[THRESHOLD_WINDOW + 1:]:
                            if FUTURE > thresholds_sd [BIN][1]:
                                m = bin[WINDOW_NO]-bin[WINDOW_NO - 1]
                                q = bin[WINDOW_NO - 1]
                                thresholds_sd [BIN][2] = (thresholds_sd [BIN][1]/m) - (q/m) + (WINDOW_NO -1)
                                break
                            WINDOW_NO += 1          
                    break
                else:
                    # If the derivative is higher than the threshold, the threshold is updated to be more sensitive next window 
                    if break_in_offset_sd + 0.1 <= 0.0: 
                        break_in_offset_sd += 0.1
                    else:
                        break_in_offset_sd = 0.0
                    # If the acutal value is higher than the threshold registred, update the threshold
                    if fixed_y < bin[THRESHOLD_WINDOW]:
                        fixed_y = bin[THRESHOLD_WINDOW]
                    
                    #Update the derivative for the next iteration
                    previous_derivative = derivative   

        return thresholds_sd

    def make_events (self, 
            set_analytic: pd.DataFrame
        ) -> (list, list):

        """
        Generates events for each bearing based on the given set of analytic data.

        Args:
        - set_analytic (DataFrame): The analytic data for all bearings.

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
            bearing_th_and_br_KL = self.evaluator_breakpoint_KL(eval_KL)
            bearing_th_and_br_SD = self.evaluator_breakpoint_SD(eval_SD)
            events_KL.append(bearing_th_and_br_KL)
            events_SD.append(bearing_th_and_br_SD)

        return events_KL, events_SD

    def calculate_L10 (self):

        base_dynamic_load_C = 12.82 * 1000
        dynamic_equivalent_load_P = 11 * 1000 #Kn From the specs of XJTU-SY dataset of this type of testset
        bearing_type_p = 3

        RPM = 37.5 * 60 #KHz From the specs of XJTU-SY dataset of this type of testset

        L10 = pow((base_dynamic_load_C / dynamic_equivalent_load_P), bearing_type_p)
        L10H = 1000000 * L10 / (60 * RPM)

        return L10H