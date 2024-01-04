import pandas as pd
import numpy as np
import statistics
import random
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import config as cfg

class DataETL:

    def __init__ (self, dataset, bootstrap):
        if dataset == "xjtu":
            self.real_bearings = cfg.N_REAL_BEARING_XJTU
            self.boot_folder_size = (2 + bootstrap) * 2
            self.total_bearings = self.real_bearings * self.boot_folder_size
            self.total_signals = cfg.N_SIGNALS_XJTU
            self.time_split = 20
            self.folder_size = self.total_bearings * self.time_split
        elif dataset == "pronostia":
            self.real_bearings = cfg.N_REAL_BEARING_PRONOSTIA
            self.boot_folder_size = (2 + bootstrap) * 2
            self.total_bearings = self.real_bearings * self.boot_folder_size
            self.total_signals = cfg.N_SIGNALS_PRONOSTIA
            self.time_split = 20
            self.folder_size = self.total_bearings * self.time_split
        self.event_detector_goal = cfg.EVENT_DETECTOR_CONFIG


    def make_surv_data_bootstrap (self, 
            covariates: dict, 
            set_boot: int, 
            info_pack: dict, 
            bootstrap: int
        ) -> (list, dict):

        """
        Transform the timeseries data into survival data.
        Prepare the survival data only for bootstrap upsampling strategy and avoid information leaking train/test.

        Args:
        - covariates (dict): Dictionary containing the covariates data.
        - set_boot (int): Number of bootstrap sets.
        - info_pack (dict): Dictionary containing information about the time-to-event.
        - bootstrap (int): The value of bootstrapped multiplier.

        Returns:
        - survival_covariates (list): List of dataframes containing the upsampled data for each group of bearings.
        - reference_value_TtE_abs (dict): Dictionary containing the reference values for each bearing of the event time with absolute values of time-to-event.
        """

        #Prepare the empty data and folders of grouped bearings to fill for avoid leaking
        row = pd.DataFrame()
        reference_value_TtE = {}
        survival_covariates = []
        type_foldering = []
        
        # Prepare the structured empty data and for grouped bearings to fill for avoid leaking
        survival_covariates, type_foldering = self.prepare_data()

        # Extract the info about the time-to-event
        for bear_num in range (1, self.total_bearings + 1, (bootstrap * 2) + 4):
            val = self.event_analyzer (bear_num, info_pack)
            reference_value_TtE.update({bear_num : val})
        
        # Set up the time slices
        moving_window = 0
        time_split = self.time_split

        # For each covariates, event or survival time take the all the time values in a column 
        for column in covariates:
            columnSeriesObj = covariates[column]
            columnSeriesObj= columnSeriesObj.dropna()

            bear_num = int(re.findall("\d?\d?\d", column)[0])
            temp_label_cov= ""
            
            # Assign the label to the covariate or take the event or survival time value
            temp_label_cov = re.split("\d_", column)[1]

            if re.findall(r"Event\b", column):
                reference_bearing_num = self.select_reference(bear_num, (bootstrap * 2) + 4)
                # If the event is negative means that event detector has no detection and need to force the censoring status
                if reference_value_TtE[reference_bearing_num] < 0:        
                    columnSeriesObj = self.event_manager(bear_num, bootstrap, self.total_bearings, force_censoring = True)
                else:
                    columnSeriesObj = self.event_manager(bear_num, bootstrap, self.total_bearings, force_censoring = False)
            elif re.findall(r"Survival_time\b", column):
                reference_bearing_num = self.select_reference(bear_num, (bootstrap * 2) + 4)
                 # After use the information about censoring transform the negative value of TtE into positive for the survival data requirements
                if reference_value_TtE[reference_bearing_num] < 0:
                    reference_value_TtE_abs = reference_value_TtE
                    reference_value_TtE_abs[reference_bearing_num] = abs(reference_value_TtE_abs[reference_bearing_num])
                    columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                else:
                    reference_value_TtE_abs = reference_value_TtE
                    columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
            
            # Actual information in subject that convert timeseries into survival data
            label= temp_label_cov
            
            # If survival time or event type take the values as is, instead for the others take the mean of time-series             
            if label == "Event" or label == "Survival_time":
                row [label]= pd.Series(columnSeriesObj).T   
            else:
                row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

            # If survival time column close the row for survival data and go to the next
            if label == "Survival_time":
                for FOLDER, bearings_in_folder in enumerate(type_foldering):
                    if bear_num in bearings_in_folder:
                        survival_covariates[FOLDER] = pd.concat([survival_covariates[FOLDER], row], ignore_index= True) 

        return survival_covariates, reference_value_TtE_abs
            
    def make_surv_data_upsampling(self, 
            covariates: dict, 
            set_boot: int, 
            info_pack: dict, 
            bootstrap: int, 
            type_correlation: str
        ) -> (list, dict):

        """
        Transform the timeseries data into survival data using boostrap and upsampling methododology.
        Upsamples the data from given time split and prepares the empty data and folders of grouped bearings to fill for avoid information leaking train/test.

        Parameters:
        - covariates (dict): Dictionary containing the covariates data.
        - set_boot (int): Number of bootstrap sets.
        - info_pack (dict): Dictionary containing the information about the time-to-event.
        - bootstrap (int): The value of bootstrapped multiplier.
        - type_correlation (str): Type of data correlation.

        Returns:
        - survival_covariates (list): List of dataframes containing the upsampled data for each group of bearings.
        - reference_value_TtE_abs (dict): Dictionary containing the reference values for each bearing of the event time with absolute values of time-to-event.
        """

        #Prepare the empty data
        row = pd.DataFrame()
        reference_value_TtE = {}
        
        # Prepare the structured empty data and for grouped bearings to fill for avoid leaking
        survival_covariates, type_foldering = self.prepare_data()

        # Extract the info about the time-to-event
        for bear_num in range (1, self.total_bearings + 1, (bootstrap * 2) + 4):
            val = self.event_analyzer (bear_num, info_pack)
            reference_value_TtE.update({bear_num : val})
        
        # Set up the time slices
        moving_window = 0
        time_split = self.time_split

        # Upsample the data from given time split
        while moving_window < time_split:
            # Set-up a condition for a very low lifetime dataset below the time_split threshold
            force_stop = False
            # For each covariates, event or survival time take the all the time values in a column 
            for column in covariates:
                columnSeriesObj = covariates[column]
                columnSeriesObj= columnSeriesObj.dropna()

                bear_num = int(re.findall("\d?\d?\d", column)[0])
                temp_label_cov= ""                    
                
                # Manage the time slicing and cases when the time split is not enough for the time-to-event (early event anomaly)
                timepoints= int(self.survival_time_manager(bear_num, set_boot, reference_value_TtE))
                time_window= int(timepoints / time_split)
                if time_window == 0:
                    if moving_window < timepoints:
                        low= moving_window
                        high= moving_window + 1
                    else:
                        force_stop= True
                else:                        
                    low= time_window * moving_window
                    high= time_window * (moving_window + 1)

                # Take the second part that contain the name of the covariate from the column name as label
                temp_label_cov = re.split("\d_", column)[1]

                if re.findall(r"Event\b", column):
                    reference_bearing_num = self.select_reference(bear_num, (bootstrap * 2) + 4)
                    # If the event is negative means that event detector has no detection and need to force the censoring status
                    if reference_value_TtE[reference_bearing_num] < 0:        
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, self.total_bearings, force_censoring = True)
                    else:
                        columnSeriesObj = self.event_manager(bear_num, bootstrap, self.total_bearings, force_censoring = False)
                elif re.findall(r"Survival_time\b", column):
                    reference_bearing_num = self.select_reference(bear_num, (bootstrap * 2) + 4)
                    # After use the information about censoring transform the negative value of TtE into positive for the survival data requirements
                    if reference_value_TtE[reference_bearing_num] < 0:
                        reference_value_TtE_abs = reference_value_TtE
                        reference_value_TtE_abs[reference_bearing_num] = abs(reference_value_TtE_abs[reference_bearing_num])
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                    else:
                        reference_value_TtE_abs = reference_value_TtE
                        columnSeriesObj = self.survival_time_manager(bear_num, set_boot, reference_value_TtE_abs)
                
                # Actual information in subject that convert timeseries into survival data
                label= temp_label_cov
                
                # If survival time or event type start to slice the time-series, instead for the others take the mean of time-series                     
                if (label == "Event" or label == "Survival_time") and force_stop == False:
                    if label == "Survival_time":
                        # If correlated start to create time-to-event relative to time slice
                        if type_correlation == "correlated" or type_correlation == "not_correlated":
                            if high < timepoints:
                                if columnSeriesObj > high:
                                    proportional_value= columnSeriesObj - high
                                else:
                                    proportional_value= columnSeriesObj                                     
                            else:
                                proportional_value= columnSeriesObj
                            row [label]= pd.Series(proportional_value).T
                        # If not correlated take the time-to-event as is
                        else:  
                            proportional_value= columnSeriesObj
                            row [label]= pd.Series(proportional_value).T  
                    else:
                        row [label]= pd.Series(columnSeriesObj).T   
                elif force_stop == False:
                    # If correlated start to slice the time-series data taking the lasts to correlate to the firsts        
                    if type_correlation == "correlated":
                        if high < timepoints: #and (time_window * (moving_window + 1)) < columnSeriesObj.size:
                            time_slice= columnSeriesObj.iloc[- (time_window * (moving_window + 1)):] #.iloc[low:high]
                        else:
                            time_slice= columnSeriesObj #.iloc[low:-1]
                    # If not correlated take the nearest slice to the time-to-event                                 
                    else:                            
                        if high < timepoints:
                            time_slice= columnSeriesObj.iloc[low:high]
                        else:
                            time_slice= columnSeriesObj.iloc[low:-1]
                    
                    # Average the covariates value to transform time-series to survival data
                    row [label]= pd.Series(np.mean(time_slice.values)).T
                
                # If survival time column close the row and save it into a proper folder for avoid leaking
                if label == "Survival_time" and force_stop == False:
                    for FOLDER, bearings_in_folder in enumerate(type_foldering):
                        if bear_num in bearings_in_folder:
                            survival_covariates[FOLDER] = pd.concat([survival_covariates[FOLDER], row], ignore_index= True)

            # Go for next window                
            moving_window += 1  
                  
        return survival_covariates, reference_value_TtE_abs

    def event_manager(self, 
            num: int, 
            bootstrap: int, 
            tot: int, 
            force_censoring: bool
        ) -> (bool):
        """
        Determines whether an event is censored or not

        Parameters:
        - num (int): The current bearing number.
        - bootstrap (int): The value of bootstrapped multiplier.
        - tot (int): The total number of events.
        - force_censoring (bool): Flag indicating whether to force censoring or not.

        Returns:
        - (bool): True if the event is not censored, False otherwise.
        """
        # The event is not determined by KL and SD so it is censored
        if force_censoring == True:
            return False

        checker = True

        # Only the two last bootstrapped bearings will be censored        
        censor_level = int((self.total_bearings / self.real_bearings) - 1)  
        for check in range(censor_level, tot + 1, (bootstrap * 2) + 4):
            if check == num or check == num - 1: 
                checker = False
                break
            else:
                checker = True
            
        if checker == False:
            return False
        else:
            return True

    def survival_time_manager (self, 
            num: int, 
            bootref: int, 
            ref: dict
        ) -> (float):

        """
        Calculates the survival time for a given bearing number adding a last randomizer from the special reference bearing.

        Parameters:
        - num (int): The bearing number.
        - bootref (DataFrame): The bootstrapped reference values.
        - ref (dict): The reference values for special bearings.

        Returns:
        - (float): The calculated survival time for the given bearing number.
        """

        # Prepare random value for the special reference bearing
        for key, value in ref.items():
            if key == num or key + 1 == num:
                return value + random.randint(-2, 2)    
        
        # Prepare the time-to-event referrement
        bootstrap= len (bootref) - 1
        tot= ((bootstrap * 2) + 4) * self.real_bearings
        boot_pack_level=  int((self.total_bearings / self.real_bearings) + 1)
        boot_pack_max= int (self.total_bearings / self.real_bearings)
        num_ref= self.total_signals
       
        # Bootstrapping + addtitional randomizator for the other normal bootstrapped/upsampled bearings
        for BEARING, check in enumerate(range(boot_pack_level, tot + (bootstrap * 2) + self.real_bearings, (bootstrap * 2) + 4)):    
            if not num >= check:
                if num== num_ref:
                    return bootref.iat[0,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)               
                elif num== num_ref + 1:
                    return bootref.iat[0,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)     
                elif num== num_ref + 2:
                    return bootref.iat[1,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)               
                elif num== num_ref + 3:
                    return bootref.iat[1,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)   
                elif num== num_ref + 4:
                    return bootref.iat[2,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 5:
                    return bootref.iat[2,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)
                elif num== num_ref + 6:
                    return bootref.iat[3,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 7:
                    return bootref.iat[3,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)
                elif num== num_ref + 8:
                    return bootref.iat[4,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 9:
                    return bootref.iat[4,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)      
                elif num== num_ref + 10:
                    return bootref.iat[5,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 11:
                    return bootref.iat[5,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)    
                elif num== num_ref + 12:
                    return bootref.iat[6,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 13:
                    return bootref.iat[6,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)    
                elif num== num_ref + 14:
                    return bootref.iat[7,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 15:
                    return bootref.iat[7,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)     
                elif num== num_ref + 16:
                    return bootref.iat[8,BEARING] + ref[check - boot_pack_max] + random.randint(-2, -1)              
                elif num== num_ref + 17:
                    # Max bootstrap 8
                    return bootref.iat[8,BEARING] + ref[check - boot_pack_max] + random.randint(1, 2)      
                
            num_ref+= (bootstrap * 2) + 4

        #Value that will be recognized if the bearing number is not in the range
        return -1

    def event_analyzer(self, 
        bear_num: int, 
        info_pack: dict
        ) -> (float):
        """
        Analyzes the event data for a specific bearing.

        Parameters:
        - bear_num (int): The number of the bearing.
        - info_pack (dict): A dictionary containing information about time-to-event for each bearing.

        Returns:
        - result (float): The analyzed event data.
        """
        
        data_kl = []
        data_sd = []

        # From info pack take the information KL and SD about time-to-event for each bearing
        for INFO_TYPE in info_pack:
            for bear_info in info_pack[INFO_TYPE][bear_num]:
                # Multiply by 10 for scaling
                cross = bear_info[2] * 10
                # Multiply by 10 for scaling
                tot_length = bear_info[3] * 10
                
                # Do not record the event if the time-to-event is 0 because there is not a breakpoint or not determined a breakline
                if INFO_TYPE == "KL" and cross != 0.0:
                    data_kl.append(cross)
                    
                # Do not record the event if the time-to-event is 0 because there is not a breakpoint or not determined a breakline
                if INFO_TYPE == "SD" and cross != 0.0:
                    data_sd.append(cross)

        # Assign a time-to-event if exist from KL or SD or end of recording if necessary
        if not data_sd:
            if data_kl:
                data_sd = data_kl
            else:
                data_sd = [-tot_length]
                print("For bearing #{}, event considered at the end of the recording".format(bear_num))
        if not data_kl:
            if data_sd:
                data_kl = data_sd

        # #Take the maximum from the evaluations
        # res = [max(data_kl), max(data_sd)]
        # res = round(statistics.mean(res), 1)

        # Take the SD as first choice and KL as second choice
        if self.event_detector_goal == "predictive_maintenance_sensitive":
            result = round(data_sd[0], 1)
        elif self.event_detector_goal == "predictive_maintenance_robust":
            if len(data_sd) > 1:
                result = round(data_sd[1], 1)
            else:
                print("For bearing #{}, event considered at the end of the recording".format(bear_num))
                result = -tot_length
        elif self.event_detector_goal == "labeler_median":
            result = round(np.median(data_sd), 1)
        elif self.event_detector_goal == "labeler_mean":
            result= round(np.mean(data_sd), 1)

        return result

    def select_reference(self, 
            bearing_num: int, 
            folder_size: int
        ) -> (int):

        """
        Selects the reference bearing for a given bearing number.

        Parameters:
        - bearing_num (int): The bearing number.
        - folder_size (int): The size of the folder.

        Returns:
        - reference (int): The reference bearing number.
        """

        folder_no = folder_size
        reference = 1

        while folder_no / bearing_num < 1.0:
            folder_no += folder_size
            reference += folder_size
        
        return reference

    def prepare_data(self
    ) -> (list, list):

        """
        Prepare the data for survival analysis.

        Returns:
        - survival_covariates (list): A list of empty DataFrames for each bearing.
        - type_foldering (list): A list of ranges representing the folders of grouped bootstrapped bearings.
        """

        survival_covariates = []
        type_foldering = []

        # Prepare the empty data and for all the bearings
        for _ in range(0, self.real_bearings, 1):
            survival_covariates.append(pd.DataFrame())

        # Prepare the folders of grouped bootstrapped bearings to fill for avoid leaking depending on the bootstrap value and consequently the folder size
        for FOLDER in range(1, self.boot_folder_size * self.real_bearings, self.boot_folder_size):
            type_foldering.append(range(FOLDER, (FOLDER + self.boot_folder_size), 1))

        return survival_covariates, type_foldering


            





