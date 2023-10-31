DATA_TYPE= ["bootstrap", "correlated", "not_correlated"]
DATASET_NAME= ["xjtu", "pronostia"]
N_BOOT= 3
N_BOOT_FOLD_XJTU= (2 + N_BOOT) * 2
N_BOOT_FOLD_UPSAMPLING= N_BOOT_FOLD_XJTU * 20
N_SIGNALS_XJTU= 2 + 1
N_BOOT_FOLD_PRONOSTIA= (2 + N_BOOT) * 2
N_SIGNALS_PRONOSTIA= 2 + 1
DATASET_PATH_XJTU= "./data/XJTU-SY/csv/"
RAW_DATA_PATH_XJTU= ["./data/XJTU-SY/35Hz12kN/", "./data/XJTU-SY/37.5Hz11kN/", "./data/XJTU-SY/40Hz10kN/"] 
N_REAL_BEARING_XJTU= 5
N_BEARING_TOT_XJTU= N_REAL_BEARING_XJTU * N_BOOT_FOLD_XJTU
DATASET_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
RAW_DATA_PATH_PRONOSTIA= ["./data/PRONOSTIA/25Hz5kN/", "./data/PRONOSTIA/27.65Hz4.2kN/", "./data/PRONOSTIA/30Hz4kN/"]  
CENSORING_LEVEL= [0.1, 0.2, 0.3]
N_REAL_BEARING_PRONOSTIA= 2
N_BEARING_TOT_PRONOSTIA= N_REAL_BEARING_PRONOSTIA * N_BOOT_FOLD_PRONOSTIA
RESULT_PATH_XJTU= "./data/XJTU-SY/results/"
SAMPLE_PATH_XJTU= "./data/XJTU-SY/csv/"
RESULT_PATH_PRONOSTIA= "./data/PRONOSTIA/results/"
SAMPLE_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
HYPER_RESULTS= "./data/logs/"

FREQUENCY_BANDS1 = {'xjtu_start': [12, 34, 71, 107, 171],
                       'xjtu_stop': [14, 36, 73, 109, 173]}
FREQUENCY_BANDS2 = {'xjtu_start': [13, 36, 76, 114, 183],
                       'xjtu_stop': [15, 38, 78, 116, 185]}
FREQUENCY_BANDS3 = {'xjtu_start': [14, 39, 82, 122, 195],
                       'xjtu_stop': [15, 41, 84, 124, 197]}
FREQUENCY_BANDS4 = {'pronostia_start': [9, 24, 89, 139, 183],
                       'pronostia_stop': [11, 26, 91, 141, 185]} 
FREQUENCY_BANDS5 = {'pronostia_start': [11, 27, 98, 153, 203],
                       'pronostia_stop': [13, 29, 100, 155, 205]}    
FREQUENCY_BANDS6 = {'pronostia_start': [12, 29, 107, 167, 216],
                       'pronostia_stop': [14, 31, 109, 169, 218]}
   
PH_EXCLUSION = {'pronostia_not_corr': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise','impulse','skew','shape','mean','max'],
                'pronostia_corr': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise','skew', 'clearence', 'mean', 'kurtosis', 'max', 'shape', 'entropy', 'impulse', 'crest', 'p2p'],
                'pronostia_boot': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise'],
                'xjtu_not_corr': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise', 'skew', 'shape', 'max', 'p2p', 'entropy', 'kurtosis'],
                'xjtu_corr': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise', 'entropy', 'p2p', 'impulse','max', 'kurtosis', 'crest', 'shape', 'skew', 'mean'],
                'xjtu_boot': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH', 'noise','max','impulse','skew']}  

PARAMS_CPH = {'alpha': 0.0001,
              'tol': 0.1, 
              'n_iter': 50}

PARAMS_CPH_RIDGE = {'alpha': 0.5,
                    'ties': 'breslow',
                    'n_iter': 100,
                    'tol': 1e-9}

PARAMS_CPH_LASSO = {'l1_ratio': 1.0,
                    'alpha_min_ratio': 0.5,
                    'fit_baseline_model': True,
                    'normalize': False,
                    'tol': 1e-7,
                    'max_iter': 100000}

PARAMS_CPH_ELASTIC = {'l1_ratio': 0.5,
                    'alpha_min_ratio': 0.1,
                    'fit_baseline_model': True,
                    'normalize': False,
                    'tol': 1e-7,
                    'max_iter': 100000}

PARAMS_RSF = {'n_estimators': 100, 
              'min_samples_split': 2, 
              'min_samples_leaf': 4, 
              'max_depth': 7}

PARAMS_GRADBOOST = {'n_estimators': 400, 
                    'min_samples_split': 2, 
                    'min_samples_leaf': 4, 
                    'max_depth': 5, 
                    'learning_rate': 0.1}

PARAMS_GRADBOOST_DART = {'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'loss': 'coxph',
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'dropout_rate': 0.2,
                        'subsample': 1.0,
                        'random_state': 0}

PARAMS_SVM = {'alpha': 1, 
              'rank_ratio': 0.8,
              'max_iter': 40, 
              'optimizer': 'avltree'}

PARAMS_DEEPSURV = {'batch_size' : 16,
                   'learning_rate' : 0.01,
                   'iters': 50}

PARAMS_DSM = {'learning_rate': 0.01, 
              'iters': 50, 
              'batch_size': 64}

PARAMS_WEIBULL = {'penalizer': 0.02, 
                  'alpha': 0.4}