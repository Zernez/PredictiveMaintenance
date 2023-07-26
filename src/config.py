N_BEARING_TOT_XJTU= 50
N_REAL_BEARING_XJTU= 5
N_BOOT_FOLD_XJTU= 10
N_SIGNALS_XJTU= 2 + 1
N_BEARING_TOT_PRONOSTIA= 20
N_REAL_BEARING_PRONOSTIA= 2
N_BOOT_FOLD_PRONOSTIA= 10
N_SIGNALS_PRONOSTIA= 2 + 1
DATASET_PATH_XJTU= "./data/XJTU-SY/csv/"
RAW_DATA_PATH_XJTU= "./data/XJTU-SY/35Hz12kN/"
DATASET_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
RAW_DATA_PATH_PRONOSTIA= "./data/PRONOSTIA/30Hz4kN/"
RESULT_PATH_XJTU= "./data/XJTU-SY/results/"
SAMPLE_PATH_XJTU= "./data/XJTU-SY/csv/"
RESULT_PATH_PRONOSTIA= "./data/PRONOSTIA/results/"
SAMPLE_PATH_PRONOSTIA= "./data/PRONOSTIA/csv/"
HYPER_RESULTS= "./data/logs/"
CPH_RESULTS= "./data/logs/Cph_results.csv"
CPHELASTIC_RESULTS= "./data/logs/CphElastic_results.csv"
CPHLASSO_RESULTS= "./data/logs/CphLasso_results.csv"
CPHRIDGE_RESULTS= "./data/logs/CphRidge_results.csv"
DEEPSURV_RESULTS= "./data/logs/DeepSurv_results.csv"
GB_RESULTS= "./data/logs/GradientBoosting_results.csv"
GB_DART_RESULTS= "./data/logs/GradientBoostingDART_results.csv"
RSF_RESULTS= "./data/logs/RSF_results.csv"
LOGLOG_RESULTS= "./data/logs/LogLogisticAFT_results.csv"
LOGNORMAL_RESULTS= "./data/logs/LogNormalAFT_results.csv"
WEIBULL_RESULTS= "./data/logs/WeibullAFT_results.csv"

PARAMS_CPH = {'alpha': 0.1,
            'ties': 'breslow',
            'n_iter': 100,
            'tol': 1e-9}

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

PARAMS_RSF = {
    'n_estimators': 100,
    'max_depth' : None,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': None,
    'random_state': 0
}

PARAMS_GRADBOOST = {'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'loss': 'coxph',
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': None,
                    'dropout_rate': 0.0,
                    'subsample': 1.0,
                    'random_state': 0}

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

PARAMS_DEEPSURV = {'batch_size' : 100,
                    'learning_rate' : 1e-3,
                    'iters': 1}

PARAMS_WEIBULL = {'alpha': 0.05,
                'penalizer': 0.0,
                'l1_ratio': 0.0,
                'fit_intercept': True}