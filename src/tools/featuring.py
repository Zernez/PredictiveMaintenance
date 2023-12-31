import os
import numpy as np 
import pandas as pd 
import random
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

        if bootstrap > 8:
            raise Exception("Too much bootstrapping, max is 8")

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

        data_total, bootstrap_total = Featuring.control_bootstrap(data, bootstrap)

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

        start = []
        stop = []
        # Load the frequency bands from config.py
        if re.findall("25Hz5kN/", dataset_path) == ['25Hz5kN/']:
            start= cfg.FREQUENCY_BANDS4['pronostia_start']
            stop= cfg.FREQUENCY_BANDS4['pronostia_stop']
        elif re.findall("27.65Hz4.2kN/", dataset_path) == ['27.65Hz4.2kN/']:
            start= cfg.FREQUENCY_BANDS5['pronostia_start']
            stop= cfg.FREQUENCY_BANDS5['pronostia_stop']
        elif re.findall("30Hz4kN/", dataset_path) == ['30Hz4kN/']:
            start= cfg.FREQUENCY_BANDS6['pronostia_start']
            stop= cfg.FREQUENCY_BANDS6['pronostia_stop']

        if bootstrap > 8:
            raise Exception("Too much bootstrapping, max is 8")

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

        data_total, bootstrap_total = Featuring.control_bootstrap(data, bootstrap)

        return data_total, bootstrap_total
    
    @classmethod
    def control_bootstrap(self, 
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
        # Bootstrap max value is 8 so the maximum distance is +-4
        random_generator_bootstrap = np.random.permutation([-4, -3, -2, -1, 1, 2, 3, 4])
        random_generator_bootstrap = random_generator_bootstrap[:bootstrap]
        data_boot = data
        data_total = []

        # Bootstrap the data stay at the random distance from the true value given permutations
        for boot_num in range(0, bootstrap + 2, 1):
            if boot_num == 0:
                data_boot.reset_index(inplace=True, drop=True)
                data_total.append(data_boot)
            elif boot_num > 0 and boot_num < bootstrap + 1:
                if random_generator_bootstrap[boot_num - 1] < 0:
                    data_boot = data
                    data_boot = data_boot.iloc[-(random_generator_bootstrap[boot_num - 1]):, :]
                else:
                    data_boot = data
                    data_aux = pd.DataFrame(data_boot[-1:].values, columns=data.columns)
                    for boot_adder in range(0, random_generator_bootstrap[boot_num - 1], 1):
                        data_boot = pd.concat([data_boot, data_aux], ignore_index=True)
                data_boot.reset_index(inplace=True, drop=True)
                data_total.append(data_boot)
            elif boot_num == bootstrap + 1:
                data_boot = data
                random_generator_bootstrap = np.append(random_generator_bootstrap, random.randint(-4, -1))
                data_boot = data_boot.iloc[-(random_generator_bootstrap[boot_num - 1]):, :]
                data_boot.reset_index(inplace=True, drop=True)
                data_total.append(data_boot)

        bootstrap_total = pd.DataFrame(random_generator_bootstrap, columns=["Bootstrap values"])

        return data_total, bootstrap_total
    
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