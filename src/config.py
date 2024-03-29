DATA_TYPE= ["bootstrap", "not_bootstrap"]
DATASET_NAME= ["xjtu", "pronostia"]
EVENT_DETECTOR_CONFIG = "labeler_max" # "predictive_maintenance_sensitive" or "predictive_maintenance_robust" or "labeler_median" or "labeler_mean"
#N_BOOT= 3
#N_BOOT_FOLD_XJTU= (2 + N_BOOT) * 2
#N_BOOT_FOLD_UPSAMPLING= N_BOOT_FOLD_XJTU * 20
N_SIGNALS_XJTU= 2 + 1
#N_BOOT_FOLD_PRONOSTIA= (2 + N_BOOT) * 2
N_SIGNALS_PRONOSTIA= 2 + 1

DATASET_PATH_XJTU= "./data/XJTU-SY/csv/"
RAW_DATA_PATH_XJTU= ["./data/XJTU-SY/35Hz12kN/", "./data/XJTU-SY/37.5Hz11kN/", "./data/XJTU-SY/40Hz10kN/"]
N_REAL_BEARING_XJTU= 5
BASE_DYNAMIC_LOAD_XJTU= 12.82
#N_BEARING_TOT_XJTU= N_REAL_BEARING_XJTU * N_BOOT_FOLD_XJTU

DATASET_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
RAW_DATA_PATH_PRONOSTIA= ["./data/PRONOSTIA/25Hz5kN/", "./data/PRONOSTIA/27.65Hz4.2kN/", "./data/PRONOSTIA/30Hz4kN/"]  
N_REAL_BEARING_PRONOSTIA= 2
BASE_DYNAMIC_LOAD_XJTU= 12.82
#N_BEARING_TOT_PRONOSTIA= N_REAL_BEARING_PRONOSTIA * N_BOOT_FOLD_PRONOSTIA

CENSORING_LEVELS= [0.25, 0.5, 0.75]
RESULTS_PATH = "./results"
PLOTS_PATH = "./plots"
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
FREQUENCY_BANDS5 = {'pronostia_start': [11, 26, 98, 153, 203],
                       'pronostia_stop': [13, 28, 100, 155, 205]}    
FREQUENCY_BANDS6 = {'pronostia_start': [12, 29, 107, 167, 216],
                       'pronostia_stop': [14, 31, 109, 169, 218]}

DATASHEET_LIFETIMES = {'xjtu_c1_b1': 123,
                       'xjtu_c1_b2': 161,
                       'xjtu_c1_b3': 158,
                       'xjtu_c1_b4': 122,
                       'xjtu_c1_b5': 52,
                       'xjtu_c2_b1': 491,
                       'xjtu_c2_b2': 161,
                       'xjtu_c2_b3': 533,
                       'xjtu_c2_b4': 42,
                       'xjtu_c2_b5': 339,
                       'xjtu_c3_b1': 2538,
                       'xjtu_c3_b2': 2496,
                       'xjtu_c3_b3': 371,
                       'xjtu_c3_b4': 1515,
                       'xjtu_c3_b5': 114
}

PH_EXCLUSION = {'xjtu_c1': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                            'FrH', 'FrpH', 'FcaH', 'noise', 'std',
                            'kurtosis', 'rms', 'crest', 'impulse'],
                'xjtu_c2': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                            'FrH', 'FrpH', 'FcaH', 'noise', 'mean',
                            'entropy', 'p2p', 'crest', 'clearence', 'impulse'],
                'xjtu_c3': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                            'FrH', 'FrpH', 'FcaH', 'noise', 'mean', 'kurtosis',
                            'crest', 'clearence', 'shape', 'impulse']}

"""
PH_EXCLUSION = {'xjtu_bootstrap_c1': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                      'FrH', 'FrpH', 'FcaH', 'noise'],
                'xjtu_bootstrap_c2': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                      'FrH', 'FrpH', 'FcaH', 'noise', 'mean',
                                      'max', 'p2p', 'clearence'],
                'xjtu_bootstrap_c3': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                      'FrH', 'FrpH', 'FcaH', 'noise', 'kurtosis',
                                      'max', 'p2p', 'clearence', 'shape'],
                'xjtu_not_correlated_c1': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                     'FrH', 'FrpH', 'FcaH', 'noise', 'skew'],
                'xjtu_not_correlated_c2': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                     'FrH', 'FrpH', 'FcaH', 'noise', 'skew',
                                     'p2p', 'clearence'],
                'xjtu_not_correlated_c3': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                     'FrH', 'FrpH', 'FcaH', 'noise', 'mean',
                                     'skew', 'kurtosis', 'entropy', 'crest',
                                     'shape', 'impulse'],
                'xjtu_correlated_c1': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                 'FrH', 'FrpH', 'FcaH', 'noise', 'kurtosis', 'shape'],
                'xjtu_correlated_c2': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                 'FrH', 'FrpH', 'FcaH', 'noise', 'clearence'],
                'xjtu_correlated_c3': ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH',
                                 'FrH', 'FrpH', 'FcaH', 'noise', 'mean',
                                 'kurtosis', 'entropy', 'rms', 'max', 'p2p',
                                 'crest', 'shape', 'impulse']}
"""

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
                   'iters': 50,
                   'layers': [32]}

PARAMS_DSM = {'batch_size' : 16,
              'learning_rate' : 0.01,
              'iters': 50,
              'layers': [32]}

PARAMS_BNN = {'layers' : [32]}

PARAMS_WEIBULL = {'penalizer': 0.02, 
                  'alpha': 0.4}
