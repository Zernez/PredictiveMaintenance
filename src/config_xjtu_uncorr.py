PARAMS_CPH = {'alpha': 0.1,
            'ties': 'breslow',
            'n_iter': 50,
            'tol': 1e-5}

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
    'max_depth' : 7,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': None,
    'random_state': 0
}

PARAMS_GRADBOOST = {'n_estimators': 200,
                        'learning_rate': 0.05,
                        'max_depth': 5,
                        'loss': 'coxph',
                        'min_samples_split': 10,
                        'min_samples_leaf': 1,
                        'max_features': None,
                        'dropout_rate': 0,
                        'subsample': 1.0,
                        'random_state': 0}

PARAMS_GRADBOOST_DART = {'n_estimators': 200,
                        'learning_rate': 0.05,
                        'max_depth': 5,
                        'loss': 'coxph',
                        'min_samples_split': 10,
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
                   'learning_rate' : 1e-3,
                   'iters': 10}

PARAMS_DSM = {'batch_size' : 64,
                   'learning_rate' : 1e-2,
                   'iters': 10}

PARAMS_WEIBULL = {'alpha': 0.4,
                'penalizer': 0.04,
                'l1_ratio': 0.0,
                'fit_intercept': True}
