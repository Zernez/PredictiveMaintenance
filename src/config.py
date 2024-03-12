N_SIGNALS_XJTU = 3
DATASET_PATH_XJTU = "./data/XJTU-SY/csv/"
RAW_DATA_PATH_XJTU = ["./data/XJTU-SY/35Hz12kN/", "./data/XJTU-SY/37.5Hz11kN/", "./data/XJTU-SY/40Hz10kN/"]
N_REAL_BEARING_XJTU = 5
BASE_DYNAMIC_LOAD_XJTU = 12.82

CENSORING_LEVELS = [0.25, 0.5, 0.75]
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
                       'xjtu_c3_b5': 114}

FREQUENCY_FTS = ['Fca','Fi','Fo','Fr','Frp','FoH', 'FiH', 'FrH', 'FrpH', 'FcaH']
PH_EXCLUSION = {'xjtu_c1': FREQUENCY_FTS + ['noise'] + ['std', 'kurtosis', 'entropy', 'rms', 'max', 'p2p', 'shape'],
                'xjtu_c2': FREQUENCY_FTS + ['noise'] + ['mean', 'skew', 'entropy', 'rms', 'max', 'crest', 'clearence', 'impulse'],
                'xjtu_c3': FREQUENCY_FTS + ['noise'] + ['mean', 'std', 'skew', 'entropy', 'rms', 'max', 'clearence']}
                
PARAMS_CPH = {'alpha': 0.0001,
              'tol': 0.1, 
              'n_iter': 50}

PARAMS_RSF = {'n_estimators': 100, 
              'min_samples_split': 2, 
              'min_samples_leaf': 4, 
              'max_depth': 7}

PARAMS_DEEPSURV = {'batch_size' : 16,
                   'learning_rate' : 0.01,
                   'iters': 50,
                   'layers': [32]}

PARAMS_DSM = {'batch_size' : 16,
              'learning_rate' : 0.01,
              'iters': 50,
              'layers': [32]}

PARAMS_BNN = {'layers' : [32]}