import pandas as pd
import numpy as np
import statistics
import random
import re
from sksurv.util import Surv
from tools.featuring import Featuring
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

class DataETL:

    def __init__(self):
        self.total_bearings= 40

    def make_surv_data_sklS (self, covariates, set_boot, info_pack, bootstrap):

        row = pd.DataFrame()
        data_cov= pd.DataFrame()
        ref_value= {}
    
        for bear_num in range (1, self.total_bearings + 1, (bootstrap * 2) + 4):
            val= self.event_analyzer (bear_num, info_pack)
            ref_value.update({bear_num : val})
        
        for column in covariates:

            columnSeriesObj = covariates[column]
            bear_num = int(re.findall("\d?\d?\d", column)[0])
            temp_label_cov= ""
            columnSeriesObj= columnSeriesObj.dropna()
            
            if re.findall(r"mean\b", column):
                temp_label_cov = "mean"
            elif re.findall(r"std\b", column):
                temp_label_cov = "std"
            elif re.findall(r"skew\b", column):
                temp_label_cov = "skew"
            elif re.findall(r"kurtosis\b", column):
                temp_label_cov = "kurtosis"
            elif re.findall(r"entropy\b", column):
                temp_label_cov = "entropy"
            elif re.findall(r"rms\b", column):
                temp_label_cov = "rms"
            elif re.findall(r"max\b", column):
                temp_label_cov = "max"
            elif re.findall(r"p2p\b", column):
                temp_label_cov = "p2p"
            elif re.findall(r"crest\b", column):
                temp_label_cov = "crest"
            elif re.findall(r"clearence\b", column):
                temp_label_cov = "clearence"
            elif re.findall(r"shape\b", column):
                temp_label_cov = "shape"
            elif re.findall(r"impulse\b", column):
                temp_label_cov = "impulse"
            elif re.findall(r"Event\b", column):
                temp_label_cov = "Event"
                columnSeriesObj = self.ev_manager (bear_num, bootstrap, self.total_bearings)
            elif re.findall(r"Survival_time\b", column):
                temp_label_cov = "Survival_time"
                columnSeriesObj = self.sur_time_manager(bear_num, set_boot, ref_value)
            
            label= temp_label_cov
            
            if label == "Event" or label == "Survival_time":
                row [label]= pd.Series(columnSeriesObj).T  
            else:
                row [label]= pd.Series(np.mean(columnSeriesObj.values)).T

            if label == "Survival_time":
                data_cov = pd.concat([data_cov, row], ignore_index= True)

        data_sa = Surv.from_dataframe("Event", "Survival_time", data_cov)

        # if data_NN == False:
        #     return data_cov.drop(["Event", "Survival_time"], axis=1), data_sa
        return data_cov, data_sa
            
    def ev_manager (self, num, bootstrap, tot):

        checker= True

        for check in range(7, tot + 1, (bootstrap * 2) + 4):
        
            if check == num or check == num - 1: 
                checker= False
                break
            else:
                checker= True
            
        if checker == False:
            return False
        else:
            return True

    def sur_time_manager (self, num, bootref, ref):

        for key, value in ref.items():
            if key == num or key + 1 == num:
                return value + random.randint(-2, 2)    
        
        bootstrap= len (bootref) - 1
        tot= ((bootstrap * 2) + 4) * 5
        num_ref= 3
        i= 0
        
        #Bootstrapping + addtitional randomizator
        for check in range(9, tot + (bootstrap * 2) + 5, (bootstrap * 2) + 4):    
            if not num >= check:
                if num== num_ref:
                    return bootref.iat[0,i] + ref[check - 8] + random.randint(-2, -1)               
                elif num== num_ref + 1:
                    return bootref.iat[0,i] + ref[check - 8] + random.randint(1, 2)     
                elif num== num_ref + 2:
                    return bootref.iat[1,i] + ref[check - 8] + random.randint(-2, -1)               
                elif num== num_ref + 3:
                    return bootref.iat[1,i] + ref[check - 8] + random.randint(1, 2)   
                elif num== num_ref + 4:
                    return bootref.iat[2,i] + ref[check - 8] + random.randint(-2, -1)              
                elif num== num_ref + 5:
                    return bootref.iat[2,i] + ref[check - 8] + random.randint(1, 2)   
                
            num_ref+= (bootstrap * 2) + 4
            i+= 1

        return -1

    def event_analyzer (self, bear_num, info_pack):
    
        lifetime_guarantee= 30
        data_kl= []
        data_sd= []

        for info in info_pack:
            for bear_info in info_pack [info][bear_num]:
                cross= bear_info [2] * 10  #10 as window
                tot_lenght= bear_info [3] * 10  #10 as window

                if info == "KL":
                    if cross > (tot_lenght)/100 * lifetime_guarantee:
                        data_kl.append(cross)
                elif info == "SD":
                    if cross > (tot_lenght)/100 * lifetime_guarantee:
                        data_sd.append(cross)

        if not data_kl:
            if data_sd:
                data_kl= data_sd
            else:
                raise Exception("Result impredictable, supervised assestment is needed")
        if not data_sd:
            if data_kl:
                data_sd= data_kl
            else:
                raise Exception("Result impredictable, supervised assestment is needed")

        res = [max (data_kl), max (data_sd)]
        res= round (statistics.mean (res), 1)

        return res

    def make_surv_data_pyS (self, covariates, set_boot, info_pack, bootstrap):
        return 0, 0
    
    def from_raw_to_csv (self, bearings):

        bootno = 2
        bearings = self.total_bearings / 5 
        j = 1

        for bearing in range (1, bearings + 1 , 1):

            dataset_path = "./data/XJTU-SY/35Hz12kN/Bearing1_" + str(bearing)
            datasets, bootstrap_val = Featuring.time_features(dataset_path, bootstrap= bootno)
            i = 1
            for dataset in datasets:
                dataset.columns= ['B' + str(j) + '_mean' , 'B' + str(j) + '_std', 'B' + str(j) + '_skew', 'B' + str(j) + '_kurtosis', 'B' + str(j) + '_entropy', 
                                'B' + str(j) + '_rms','B' + str(j) + '_max', 'B' + str(j) + '_p2p', 'B' + str(j) + '_crest', 'B' + str(j) + '_clearence', 
                                'B' + str(j) + '_shape', 'B' + str(j) + '_impulse', 'B' + str(j) + '_freq_band_1', 'B' + str(j) + '_freq_band_2', 'B' + str(j) + '_freq_band_3', 
                                'B' + str(j) + '_freq_band_4', 'B' + str(j) + '_freq_band_5', 'B' + str(j) + '_Event', 'B' + str(j) + '_Survival_time',
                                'B' + str(j + 1) + '_mean', 'B' + str(j + 1) + '_std', 'B' + str(j + 1) + '_skew', 'B' + str(j + 1) + '_kurtosis', 'B' + str(j + 1) + '_entropy', 
                                'B' + str(j + 1) +'_rms', 'B' + str(j + 1) + '_max', 'B' + str(j + 1) + '_p2p', 'B' + str(j + 1) + '_crest' , 'B' + str(j + 1) + '_clearence', 
                                'B' + str(j + 1) + '_shape','B' + str(j + 1) + '_impulse', 'B' + str(j + 1) + '_freq_band_1' , 'B' + str(j + 1) + '_freq_band_2', 'B' + str(j + 1) + '_freq_band_3', 
                                'B' + str(j + 1) + '_freq_band_4' , 'B' + str(j + 1) + '_freq_band_5', 'B' + str(j + 1) + '_Event', 'B' + str(j + 1) + '_Survival_time']
                    
                dataname= "./data/XJTU-SY/csv/Bearing1_" + str(bearing) + "_" + str(i) + "_timefeature.csv"
                dataset.to_csv(dataname, index= False)
                i += 1
                j += 2

            dataname= "./data/XJTU-SY/csv/Bearing1_" + str(bearing) + "_bootstrap.csv"
            bootstrap_val.to_csv(dataname, index= False)

    def format_main_data (self, T1, train, test):

        # Make data split

        ti_y_df= T1[0].iloc[train, -2:]
        cvi_y_df= T1[0].iloc[test, -2:]

        ti_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)
        cvi_y_df.rename(columns = {'Event':'event', 'Survival_time':'time'}, inplace = True)

        ti_X = T1[0].iloc[train, :-2]
        ti_y = T1[1][train]
        cvi_X = T1[0].iloc[test, :-2]
        cvi_y = T1[1][test]
#        features = ti_X.columns

        # Apply scaling
        # scaler = StandardScaler()
        # scaler.fit(ti_X)
        # ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        # cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)


        #End ETL normal data and start ETL for NN

        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_y_df.reset_index(inplace= True, drop=True)
        cvi_y_df.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X, ti_y_df)
        cvi_NN = (cvi_X, cvi_y_df)

        return ti, cvi, ti_NN, cvi_NN
    
    def centering_main_data (self, ti, cvi, ti_NN, cvi_NN):

        # Make data split 
        ti_X = ti[0]
        ti_y = ti[1]
        cvi_X = cvi[0]
        cvi_y = cvi[1]
        ti_X_NN = ti_NN[0]
        ti_y_NN = ti_NN[1]
        cvi_X_NN = cvi_NN[0]
        cvi_y_NN = cvi_NN[1]
        features = list(ti_X.columns)

        # Apply scaling
        scaler = StandardScaler()

        scaler.fit(ti_X)
        ti_X = pd.DataFrame(scaler.transform(ti_X), columns=features)
        cvi_X = pd.DataFrame(scaler.transform(cvi_X), columns=features)
        ti_X_NN = pd.DataFrame(scaler.transform(ti_X_NN), columns=features)
        cvi_X_NN = pd.DataFrame(scaler.transform(cvi_X_NN), columns=features)

        ti_X.reset_index(inplace= True, drop=True)
        cvi_X.reset_index(inplace= True, drop=True)
        ti_X_NN.reset_index(inplace= True, drop=True)
        cvi_X_NN.reset_index(inplace= True, drop=True)

        # Collect splits
        ti = (ti_X, ti_y)
        cvi = (cvi_X, cvi_y)
        ti_NN = (ti_X_NN, ti_y_NN)
        cvi_NN = (cvi_X_NN, cvi_y_NN)

        return ti, cvi, ti_NN, cvi_NN
    
    def format_centering_NN_data (self, T1NN, T2NN, ti_y_df, cvi_y_df, TvalNN):

        ti_X_NN = pd.concat([T1NN[0], ti_y_df], axis=1)
        cvi_X_NN = pd.concat([T2NN[0], cvi_y_df], axis=1)
        ti_X_val_NN= TvalNN[0]


        features = T1NN[0].columns

        cols_standardize = list(features)
        cols_leave = []

        scaler= StandardScaler()

        standardize = [([col], scaler) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)

        x_train_ti = x_mapper.fit_transform(ti_X_NN).astype('float32')
        x_train_cvi = x_mapper.fit_transform(cvi_X_NN).astype('float32')       
        x_val = x_mapper.transform(ti_X_val_NN).astype('float32')

        get_target = lambda df: (df['Survival_time'].values, df['Event'].values)
        y_ti_NN = get_target(ti_X_NN)
        y_val = get_target(ti_X_val_NN)

        durations_test, events_test = get_target(cvi_X_NN)       

        ti_NN = x_train_ti
        cvi_NN = x_train_cvi
        val_NN= x_val, y_val

        return ti_NN , y_ti_NN, cvi_NN, durations_test, events_test, val_NN