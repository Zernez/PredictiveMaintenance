"""
This file has all configurations for doing the data preprocessing and all the relevant locations to read / write from.
"""

PARAMS_CPH = {
    'alpha': 0.1,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

PARAMS_CPH_RIDGE = {
    'alpha': 0.5,
    'ties': 'breslow',
    'n_iter': 100,
    'tol': 1e-9
}

PARAMS_CPH_LASSO = {
    'l1_ratio': 1.0,
    'alpha_min_ratio': 0.5,
    'fit_baseline_model': True,
    'normalize': False,
    'tol': 1e-7,
    'max_iter': 100000
}

PARAMS_CPH_ELASTIC = {
    'l1_ratio': 0.5,
    'alpha_min_ratio': 0.1,
    'fit_baseline_model': True,
    'normalize': False,
    'tol': 1e-7,
    'max_iter': 100000
    }

PARAMS_EXT = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': None
}

PARAMS_RSF = {
    'n_estimators': 100,
    'max_depth' : None,
    'n_jobs': -1,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': None,
    'random_state': 0
}

PARAMS_COXBOOST = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'dropout_rate': 0.0,
    'subsample': 1.0,
    'random_state': 0
}

PARAMS_XGB_LINEAR = {
    'n_estimators': 100,
    'objective': 'survival:cox',
    'learning_rate': 0.3,
    'base_score': 0.5,
    'booster':'gblinear',
    'random_state': 0,
}

PARAMS_XGB_GBTREE = {
    'n_estimators': 100,
    'objective': 'survival:cox',
    'tree_method': 'gpu_hist',
    'learning_rate': 0.3,
    'base_score': 0.5,
    'max_depth': 6,
    'booster':'gbtree',
    'subsample': 1,
    'min_child_weight': 1,
    'colsample_bynode':1,
    'random_state': 0
}

PARAMS_XGB_DART = {
    'n_estimators': 100,
    'objective': 'survival:cox',
    'tree_method': 'gpu_hist',
    'learning_rate': 0.3,
    'base_score': 0.5,
    'max_depth': 6,
    'booster':'dart',
    'subsample': 1,
    'min_child_weight': 1,
    'colsample_bynode':1,
    'random_state': 0
}

PARAMS_WEIBULL = {
    'alpha': 0.05,
    'penalizer': 0.0,
    'l1_ratio': 0.0,
    'fit_intercept': True
}

PARAMS_XGB_CLF = {
    "n_estimators": 100,
    "booster": "gbtree",
    "max_depth": 3,
    "gamma": 0,
    "colsample_bytree": 0.5,
    "min_child_weight": 1,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "learning_rate": 0.1,
    "subsample": 1,
    "base_score": 0.5,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
    "random_state": 0
}

PARAMS_RF_CLF = {
    'n_estimators': 100,
    'random_state': 0
}