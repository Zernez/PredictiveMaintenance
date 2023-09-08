PARAMS_CPH = {'tol': 0.1, 
              'n_iter': 100}

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

PARAMS_DEEPSURV = {'learning_rate': 0.01, 
                   'iters': 50, 
                   'batch_size': 16}

PARAMS_DSM = {'learning_rate': 0.001, 
              'iters': 50, 
              'batch_size': 32}

PARAMS_WEIBULL = {'penalizer': 0.02, 
                  'alpha': 0.4}
