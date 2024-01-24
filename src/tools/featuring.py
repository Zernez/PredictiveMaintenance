import os
import numpy as np 
import pandas as pd 
import random
import math
import re
from scipy.signal import hilbert
from scipy.stats import entropy
import config as cfg

class Featuring:

    def __init__ (self):
        pass

    @staticmethod
    def time_features_xjtu(
            dataset_path: str,
            bootstrap: int = 0
        ) -> (pd.DataFrame, pd.DataFrame):

        """
        Extracts all time and frequency features in timeseries modality from the XJTU dataset.

        Args:
        - dataset_path (str): The path to the dataset.
        - bootstrap (int): The number of bootstrapping iterations. Default is 0.

        Returns:
        - data_total (DataFrame): A dataframe containing the features extracted from the dataset in timeseries type.
        - bootstrap_total (DataFrame): Contain the generated information of the bootstrap values that will be used for construct the event data.
        """

        # All feature names
        features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse',
                'FoH', 'FiH', 'FrH', 'FrpH', 'FcaH','Fo', 'Fi', 'Fr', 'Frp', 'Fca', 'noise', 'Event', 'Survival_time']
        # Bearing names from use x and y data as B1 and B2
        cols2 = ['B1', 'B2']

        start = []
        stop = []

        start, stop = Featuring.calculate_frequency_bands(dataset_path)

        #From the specs of XJTU dataset 25.6 kHz
        Fsamp = 25600

        Ts = 1/Fsamp
        
        columns = [c+'_'+tf for c in cols2 for tf in features]
        data = pd.DataFrame(columns=columns)
        
        for filename in os.listdir(dataset_path):
            
            raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep=',')

            timeline= int(re.sub('.csv*', '', filename))

            # Time features
            mean_abs = np.array(raw_data.abs().mean())
            std = np.array(raw_data.std())
            skew = np.array(raw_data.skew())
            kurtosis = np.array(raw_data.kurtosis())
            entropy = Featuring.calculate_entropy(raw_data)
            rms = np.array(Featuring.calculate_rms(raw_data))
            max_abs = np.array(raw_data.abs().max())
            p2p = Featuring.calculate_p2p(raw_data)
            crest = max_abs/rms
            clearence = np.array(Featuring.calculate_clearence(raw_data))
            shape = rms / mean_abs
            impulse = max_abs / mean_abs
            
            # FFT with Hilbert features
            h_signal = np.abs(hilbert(raw_data))
            N = len(h_signal)
            fftH = np.abs(np.fft.fft(h_signal) /len (h_signal))
            fftH= 2 * fftH [0:int(N/2+1)]
            fftH[0]= fftH[0] / 2
            
            # FFT without Hilbert
            raw_signal = np.abs(raw_data)
            N = len(raw_signal)
            fft = np.abs(np.fft.fft(raw_signal) /len (raw_signal))
            fft= 2 * fft [0:int(N/2+1)]
            fft[0]= fft[0] / 2

            # Excluding continuous representation (Hilbert)
            start_freq_interested = start
            end_freq_interested = stop
            fft_nH = [None] * 5
            i = 0
            for index_s in start_freq_interested:
                index_e= end_freq_interested[i]
                temp_mean = fftH [index_s : index_e]
                fft_nH [i]= np.mean(temp_mean, axis= 0) 
                i += 1

            # Excluding continuous representation (NoHilbert)
            start_freq_interested = start
            end_freq_interested = stop
            fft_n = [None] * 5
            i = 0
            for index_s in start_freq_interested:
                index_e= end_freq_interested[i]
                temp_mean = fft [index_s : index_e]
                fft_n [i]= np.mean(temp_mean, axis= 0) 
                i += 1
            
            # Noise feature
            noise = np.mean(fft, axis= 0) 

            # Setup dataframe index and columns names for each feature 
            mean_abs = pd.DataFrame(mean_abs.reshape(1,2), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,2), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,2), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,2), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,2), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,2), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,2), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,2), columns=[c+'_p2p' for c in cols2])
            crest = pd.DataFrame(crest.reshape(1,2), columns=[c+'_crest' for c in cols2])
            clearence = pd.DataFrame(clearence.reshape(1,2), columns=[c+'_clearence' for c in cols2])
            shape = pd.DataFrame(shape.reshape(1,2), columns=[c+'_shape' for c in cols2])
            impulse = pd.DataFrame(impulse.reshape(1,2), columns=[c+'_impulse' for c in cols2])
            fft_1 = pd.DataFrame(fft_nH [0].reshape(1,2), columns=[c+'_FoH' for c in cols2])
            fft_2 = pd.DataFrame(fft_nH [1].reshape(1,2), columns=[c+'_FiH' for c in cols2])
            fft_3 = pd.DataFrame(fft_nH [2].reshape(1,2), columns=[c+'_FrH' for c in cols2])
            fft_4 = pd.DataFrame(fft_nH [3].reshape(1,2), columns=[c+'_FrpH' for c in cols2])
            fft_5 = pd.DataFrame(fft_nH [4].reshape(1,2), columns=[c+'_FcaH' for c in cols2])
            fft_6 = pd.DataFrame(fft_n [0].reshape(1,2), columns=[c+'_Fo' for c in cols2])
            fft_7 = pd.DataFrame(fft_n [1].reshape(1,2), columns=[c+'_Fi' for c in cols2])
            fft_8 = pd.DataFrame(fft_n [2].reshape(1,2), columns=[c+'_Fr' for c in cols2])
            fft_9 = pd.DataFrame(fft_n [3].reshape(1,2), columns=[c+'_Frp' for c in cols2])
            fft_10 = pd.DataFrame(fft_n [4].reshape(1,2), columns=[c+'_Fca' for c in cols2])
            noise = pd.DataFrame(noise.reshape(1,2), columns=[c+'_noise' for c in cols2])
            event = pd.DataFrame(np.array([False, False]).reshape(1,2), columns=[c+'_Event' for c in cols2])
            survt = pd.DataFrame(np.array([0, 0]).reshape(1,2), columns=[c+'_Survival_time' for c in cols2])

            mean_abs.index = [timeline]
            std.index = [timeline]
            skew.index = [timeline]
            kurtosis.index = [timeline]
            entropy.index = [timeline]
            rms.index = [timeline]
            max_abs.index = [timeline]
            p2p.index = [timeline]
            crest.index = [timeline]
            clearence.index = [timeline]
            shape.index = [timeline]
            impulse.index = [timeline]
            fft_1.index = [timeline]
            fft_2.index = [timeline]
            fft_3.index = [timeline] 
            fft_4.index = [timeline]
            fft_5.index = [timeline]
            fft_6.index = [timeline]
            fft_7.index = [timeline]
            fft_8.index = [timeline] 
            fft_9.index = [timeline]
            fft_10.index = [timeline]
            noise.index = [timeline]
            event.index = [timeline]
            survt.index = [timeline]      
            
            # Concat and create the master dataframe
            merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse, 
                            fft_1, fft_2, fft_3, fft_4, fft_5, fft_6, fft_7, fft_8, fft_9, fft_10, noise, event, survt], axis=1)
            data = pd.concat([data,merge])

        cols = [c+'_'+tf for c in cols2 for tf in features]
        data = data[cols]
        data.sort_index(inplace= True)
        data.reset_index(inplace= True, drop=True)  

        if bootstrap > 0:
            data_total, bootstrap_total = Featuring.control_bootstrap(data, bootstrap)
        else:
            data_total = [data]
            bootstrap_total = pd.DataFrame()

        return data_total, bootstrap_total

    @staticmethod
    def time_features_pronostia(
            dataset_path: str, 
            bootstrap: int = 0
        ) -> (pd.DataFrame, pd.DataFrame):

        """
        Extracts all time and frequency features in timeseries modality from the PRONOSTIA dataset.

        Args:
        - dataset_path (str): The path to the dataset.
        - bootstrap (int): The number of bootstrapping iterations. Default is 0.

        Returns:
        - data_total (DataFrame): A dataframe containing the features extracted from the dataset in timeseries type.
        - bootstrap_total (DataFrame): Contain the generated information of the bootstrap values that will be used for construct the event data.
        """

        # All feature names
        features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse',
                'FoH', 'FiH', 'FrH', 'FrpH', 'FcaH','Fo', 'Fi', 'Fr', 'Frp', 'Fca', 'noise', 'Event', 'Survival_time']
        # Bearing names from use x and y data as B1 and B2
        cols2 = ['B1', 'B2']

        start, stop = Featuring.calculate_frequency_bands(dataset_path)

        #From the specs of PRONOSTIA dataset 25.6 kHz
        Fsamp = 25600

        Ts = 1/Fsamp
        
        columns = [c+'_'+tf for c in cols2 for tf in features]
        data = pd.DataFrame(columns=columns)
        timeline = 1
        
        for filename in os.listdir(dataset_path):
        
            raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep=',')
            
            raw_data.drop(raw_data.columns[[0, 1, 2, 3]], axis=1, inplace=True)

            # Time features
            mean_abs = np.array(raw_data.abs().mean())
            std = np.array(raw_data.std())
            skew = np.array(raw_data.skew())
            kurtosis = np.array(raw_data.kurtosis())
            entropy = Featuring.calculate_entropy(raw_data)
            rms = np.array(Featuring.calculate_rms(raw_data))
            max_abs = np.array(raw_data.abs().max())
            p2p = Featuring.calculate_p2p(raw_data)
            crest = max_abs/rms
            clearence = np.array(Featuring.calculate_clearence(raw_data))
            shape = rms / mean_abs
            impulse = max_abs / mean_abs

            # FFT with Hilbert features
            h_signal = np.abs(hilbert(raw_data))
            N = len(h_signal)
            fftH = np.abs(np.fft.fft(h_signal) /len (h_signal))
            fftH= 2 * fftH [0:int(N/2+1)]
            fftH[0]= fftH[0] / 2

            # FFT features without Hilbert
            raw_signal = np.abs(raw_data)
            N = len(raw_signal)
            fft = np.abs(np.fft.fft(raw_signal) /len (raw_signal))
            fft= 2 * fft [0:int(N/2+1)]
            fft[0]= fft[0] / 2

            # Excluding continuous representation (Hilbert)
            start_freq_interested = start
            end_freq_interested = stop
            fft_nH = [None] * 5
            i = 0
            for index_s in start_freq_interested:
                index_e= end_freq_interested[i]
                temp_mean = fftH [index_s : index_e]
                fft_nH [i]= np.mean(temp_mean, axis= 0) 
                i += 1

            # Excluding continuous representation (NoHilbert)
            start_freq_interested = start
            end_freq_interested = stop
            fft_n = [None] * 5
            i = 0
            for index_s in start_freq_interested:
                index_e= end_freq_interested[i]
                temp_mean = fft [index_s : index_e]
                fft_n [i]= np.mean(temp_mean, axis= 0) 
                i += 1
            
            # Noise feature
            noise = np.mean(fft, axis= 0) 

            # Setup dataframe index and columns names for each feature
            mean_abs = pd.DataFrame(mean_abs.reshape(1,2), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,2), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,2), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,2), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,2), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,2), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,2), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,2), columns=[c+'_p2p' for c in cols2])
            crest = pd.DataFrame(crest.reshape(1,2), columns=[c+'_crest' for c in cols2])
            clearence = pd.DataFrame(clearence.reshape(1,2), columns=[c+'_clearence' for c in cols2])
            shape = pd.DataFrame(shape.reshape(1,2), columns=[c+'_shape' for c in cols2])
            impulse = pd.DataFrame(impulse.reshape(1,2), columns=[c+'_impulse' for c in cols2])
            fft_1 = pd.DataFrame(fft_nH [0].reshape(1,2), columns=[c+'_FoH' for c in cols2])
            fft_2 = pd.DataFrame(fft_nH [1].reshape(1,2), columns=[c+'_FiH' for c in cols2])
            fft_3 = pd.DataFrame(fft_nH [2].reshape(1,2), columns=[c+'_FrH' for c in cols2])
            fft_4 = pd.DataFrame(fft_nH [3].reshape(1,2), columns=[c+'_FrpH' for c in cols2])
            fft_5 = pd.DataFrame(fft_nH [4].reshape(1,2), columns=[c+'_FcaH' for c in cols2])
            fft_6 = pd.DataFrame(fft_n [0].reshape(1,2), columns=[c+'_Fo' for c in cols2])
            fft_7 = pd.DataFrame(fft_n [1].reshape(1,2), columns=[c+'_Fi' for c in cols2])
            fft_8 = pd.DataFrame(fft_n [2].reshape(1,2), columns=[c+'_Fr' for c in cols2])
            fft_9 = pd.DataFrame(fft_n [3].reshape(1,2), columns=[c+'_Frp' for c in cols2])
            fft_10 = pd.DataFrame(fft_n [4].reshape(1,2), columns=[c+'_Fca' for c in cols2])
            noise = pd.DataFrame(noise.reshape(1,2), columns=[c+'_noise' for c in cols2])
            event = pd.DataFrame(np.array([False, False]).reshape(1,2), columns=[c+'_Event' for c in cols2])
            survt = pd.DataFrame(np.array([0, 0]).reshape(1,2), columns=[c+'_Survival_time' for c in cols2])

            mean_abs.index = [timeline]
            std.index = [timeline]
            skew.index = [timeline]
            kurtosis.index = [timeline]
            entropy.index = [timeline]
            rms.index = [timeline]
            max_abs.index = [timeline]
            p2p.index = [timeline]
            crest.index = [timeline]
            clearence.index = [timeline]
            shape.index = [timeline]
            impulse.index = [timeline]
            fft_1.index = [timeline]
            fft_2.index = [timeline]
            fft_3.index = [timeline] 
            fft_4.index = [timeline]
            fft_5.index = [timeline]
            fft_6.index = [timeline]
            fft_7.index = [timeline]
            fft_8.index = [timeline] 
            fft_9.index = [timeline]
            fft_10.index = [timeline]
            noise.index = [timeline]
            event.index = [timeline]
            survt.index = [timeline]      
            
            # Concat and create the master dataframe
            merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse, 
                            fft_1, fft_2, fft_3, fft_4, fft_5, fft_6, fft_7, fft_8, fft_9, fft_10, noise, event, survt], axis=1)
            data = pd.concat([data,merge])

            timeline += 1

        cols = [c+'_'+tf for c in cols2 for tf in features]
        data = data[cols]
        data.sort_index(inplace= True)
        data.reset_index(inplace= True, drop=True)  

        if bootstrap > 0:
            data_total, bootstrap_total = Featuring.control_bootstrap(data, bootstrap)
        else:
            data_total = [data]
            bootstrap_total = pd.DataFrame()

        return data_total, bootstrap_total

    @classmethod
    def control_bootstrap (self, 
            data: pd.DataFrame, 
            bootstrap: int = 0
        ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        """
        Add information about bootstrap in a separate dataframe by generating random permutations and creating bootstrapped samples.
        In a second step, the survival dataset will be created by adding the bootstrap event values to the original value.

        Parameters:
        - data: The input data to be bootstrapped.
        - bootstrap: The number of bootstrap multiplier.

        Returns:
        - data_total (DataFrame): Contain the generated information of timeseries data.
        - bootstrap_total (DataFrame): Contain the generated information of the bootstrap values.
        """
        
        # Create a random normal distribution of samples for tte based on the bootstrap multiplier multiplied by 2 as the channel x and y considered separately
        tte_random_generator = self.create_normal_random_samples(bootstrap * 2)

        data_boot = data
        data_total = []

        # Add the original data
        data_boot.reset_index(inplace=True, drop=True)
        data_total.append(data_boot)

        # Deviate from the original value by 3% of the standard deviation
        std_percentage = 0.03

        # Add the bootstrapped data based on the random normal distribution tte
        for boot_num in range(0, bootstrap, 1):
            data_boot = data

            # How many times we have to pick random numbers as distance from the new target
            distance = math.ceil(abs(tte_random_generator[boot_num]))

            while distance > 0:
                new_time_row = pd.Series(dtype= np.float64)  
                for column_name, column_data in data_boot.iteritems():
                    if re.split("\d_", column_name)[1] != "Event" and re.split("\d_", column_name)[1] != "Survival_time":
                        
                        standard_deviation = column_data.std() * std_percentage
                        # Create a random normal distribution of samples for covariates for each column and each step to cover
                        covariate_random_item = self.create_covariates_random_samples(standard_deviation, column_data[-1:].values[0])
                        new_time_row [column_name]= covariate_random_item
                    else:

                        # No need to create random samples for event and survival time
                        new_time_row[column_name]= column_data[0]
                
                # Add the new row to the dataframe
                data_boot = data_boot.append(new_time_row, ignore_index=True)
                distance -= 1
            
            # Add the new bootstrapped data to the total
            data_boot.reset_index(inplace=True, drop=True)
            data_total.append(data_boot)

        bootstrap_total = pd.DataFrame(tte_random_generator, columns=["Bootstrap values"])

        return data_total, bootstrap_total

    @classmethod 
    def create_normal_random_samples (self, 
             min_samples: int = 0
        ) -> np.array:

        """
        Generate an array of normal random samples.

        Parameters:
        - min_samples (int): The minimum number of samples required. Default is 0.

        Returns:
        - np.array: An array of normal distributed random samples.
        """

        # Set the size of the sample you want in the normal distribution
        sample_size = 100

        # Set the mean and standard deviation of the normal distribution
        mean = 0
        std_dev = sample_size * 0.03

        # Set the resolution as number of decimal places
        resolution = 1

        # Generate random samples from a normal distribution
        random_samples = np.random.normal(mean, std_dev, sample_size)

        # Round the samples to the desired resolution
        random_samples = np.round(random_samples, resolution)

        # Ensure uniqueness of samples and avoid no deviation samples
        unique_random_samples = np.unique(random_samples)
        unique_random_samples = unique_random_samples[unique_random_samples != 0]

        # Check if the number of samples is less than the minimum
        while len(unique_random_samples) < min_samples:
            # Generate additional samples to meet the minimum requirement
            additional_samples = np.random.normal(mean, std_dev, min_samples - len(unique_random_samples))
            
            # Round the additional samples to the desired resolution
            additional_samples = np.round(additional_samples, resolution)
            
            # Ensure uniqueness of additional samples and avoid no deviation samples
            additional_samples = np.unique(additional_samples)
            additional_samples = additional_samples[additional_samples != 0]

            np.concatenate((unique_random_samples, additional_samples), axis= None)

        # Take the first min_samples to ensure the desired minimum number of samples
        random_generator = unique_random_samples[:min_samples]

        return random_generator

    @classmethod 
    def create_covariates_random_samples(self,
            std_dev: float, 
            ref: float, 
        ) -> float:

        """
        Generate random samples for one column and one step in the given covariates DataFrame.

        Parameters:
            std_dev (DataFrame): The standard deviation of the feature though the whole time
            ref (float): The reference value used as the mean for the normal distribution.

        Returns:
            float: Random samples generated for one column in the covariates DataFrame.
        """

        # Set the mean and standard deviation of the normal distribution
        mean = ref

        # Set the size of the sample you want in the normal distribution
        sample_size = 50

        # Set how many sample you want to generate
        min_samples = 1

        # Generate random samples from a normal distribution
        random_samples = np.random.normal(mean, std_dev, sample_size)

        # Ensure uniqueness of samples
        unique_random_samples = np.unique(random_samples)

        # Check if the number of samples is less than the minimum
        while len(unique_random_samples) < min_samples:
            # Generate additional samples to meet the minimum requirement
            additional_samples = np.random.normal(mean, std_dev, min_samples - len(unique_random_samples))

            # Round the additional samples to the desired resolution
            additional_samples = np.round(additional_samples, resolution)

            # Ensure uniqueness of additional samples
            additional_samples = np.unique(additional_samples)

            np.concatenate((unique_random_samples, additional_samples), axis= None)

        # Take the first and only min_samples
        random_generator = unique_random_samples[:min_samples][0]

        return random_generator

    @classmethod
    def calculate_frequency_bands (self,
        dataset_path:  str
    ) -> list:

        """
        Find the frequency bands of the dataset from start to end of each.
        """

        start = []
        stop = []

        # Load the frequency bands from config.py
        if re.findall("35Hz12kN/", dataset_path) == ['35Hz12kN/']:
            start= cfg.FREQUENCY_BANDS1['xjtu_start']
            stop= cfg.FREQUENCY_BANDS1['xjtu_stop']
        elif re.findall("37.5Hz11kN/", dataset_path) == ['37.5Hz11kN/']:
            start= cfg.FREQUENCY_BANDS2['xjtu_start']
            stop= cfg.FREQUENCY_BANDS2['xjtu_stop']
        elif re.findall("40Hz10kN/", dataset_path) == ['40Hz10kN/']:
            start= cfg.FREQUENCY_BANDS3['xjtu_start']
            stop= cfg.FREQUENCY_BANDS3['xjtu_stop']

        return start, stop
    
    def calculate_rms (
            df: pd.DataFrame
        ) -> list:

        result = []
        for col in df:
            r = np.sqrt((df[col]**2).sum() / len(df[col]))
            result.append(r)
        return result

    def calculate_p2p (
            df: pd.DataFrame
        ) -> np.array:

        return np.array(df.max().abs() + df.min().abs())
    
    def calculate_entropy (
            df: pd.DataFrame
        ) -> np.array:

        ent = []
        for col in df:
            ent.append(entropy(pd.cut(df[col], 500).value_counts()))
        return np.array(ent)
    
    def calculate_clearence (
        df: pd.DataFrame
        ) -> list:

        result = []
        for col in df:
            r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
            result.append(r)
        return result