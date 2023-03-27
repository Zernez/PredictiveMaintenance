from abc import ABC, abstractmethod
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import config as cfg
from lifelines import WeibullAFTFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter

class BaseRegressor(ABC):
    """
    Base class for regressors.
    """
    def __init__(self):
        """Initilizes inputs and targets variables."""

    @abstractmethod
    def make_model(self, params=None):
        """
        This method is an abstract method to be implemented
        by a concrete classifier. Must return a sklearn-compatible
        estimator object implementing 'fit'.
        """

    @abstractmethod
    def get_hyperparams(self):
        """Method"""

    @abstractmethod
    def get_best_hyperparams(self):
        """Method"""

    def get_tuneable_params(self):
        return self.get_hyperparams()

    def get_best_params(self):
        return self.get_best_hyperparams()

    def get_estimator(self, params=None):
        return self.make_model(params)

class Cph(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_CPH
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    def get_hyperparams(self):
        return {
            'n_iter': [50, 100],
            'tol': [1e-1, 1e-5, 1e-9]
        }
    def get_best_hyperparams(self):
        return {'tol': 0.1, 'n_iter': 50}

class CphRidge(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_CPH_RIDGE
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    def get_hyperparams(self):
        return {
            'n_iter': [50, 100],
            'tol': [1e-1, 1e-5, 1e-9]
        }
    def get_best_hyperparams(self):
        return {'tol': 0.1, 'n_iter': 50}

class CphLasso(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_CPH_LASSO
        if params:
            model_params.update(params)
        return CoxnetSurvivalAnalysis(**model_params)
    def get_hyperparams(self):
        return {
            'n_alphas': [10, 50, 100],
            'normalize': [True, False],
            'tol': [1e-1, 1e-5, 1e-7],
            'max_iter': [100000]
        }
    def get_best_hyperparams(self):
        return {'tol': 1e-05, 'normalize': True,
                'n_alphas': 10, 'max_iter': 100000}

class CphElastic(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_CPH_ELASTIC
        if params:
            model_params.update(params)
        return CoxnetSurvivalAnalysis(**model_params)
    def get_hyperparams(self):
        return {
            'n_alphas': [10, 50, 100],
            'normalize': [True, False],
            'tol': [1e-1, 1e-5, 1e-7],
            'max_iter': [100000]
        }
    def get_best_hyperparams(self):
        return {'tol': 1e-05, 'normalize': True,
                'n_alphas': 10, 'max_iter': 100000}

class RSF(BaseRegressor):
    def make_model(self, params=None):
        model_params = {'random_state': 0}
        if params:
            model_params.update(params)
        return RandomSurvivalForest(**model_params)
    def get_hyperparams(self):
        return {
            'max_depth': [1, 2, 3],
            'n_estimators': [25, 50, 75, 100, 200],
            'min_samples_split': [200, 400, 800],
            'min_samples_leaf': [100, 200, 400]
        }
    def get_best_hyperparams(self):
        return  {'n_estimators': 100, 'min_samples_split': 400,
                 'min_samples_leaf': 200, 'max_depth': 3}

class CoxBoost(BaseRegressor):
    def make_model(self, params=None):
        model_params = {'random_state': 0, 'subsample': 0.8}
        if params:
            model_params.update(params)
        return GradientBoostingSurvivalAnalysis(**model_params)
    def get_hyperparams(self):
        return {
            'max_depth': [1, 2, 3],
            'n_estimators': [25, 50, 75, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [200, 400, 800],
            'min_samples_leaf': [100, 200, 400]
        }
    def get_best_hyperparams(self):
        return {'n_estimators': 25, 'min_samples_split': 400,
                'min_samples_leaf': 100, 'max_depth': 3,
                'learning_rate': 0.05}

class XGBLinear(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_XGB_LINEAR
        if params:
            model_params.update(params)
        return xgb.XGBRegressor(**model_params)

    def get_hyperparams(self):
        return {
            'n_estimators': [25, 50, 75, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
        }
    def get_best_hyperparams(self):
        return {'n_estimators': 50, 'learning_rate': 0.01}

class XGBTree(BaseRegressor):
    def make_model(self, params=None):
        model_params = {'objective':'survival:cox', 'tree_method':'hist',
                        'booster':'gbtree', 'subsample':0.8, 'random_state':0}
        if params:
            model_params.update(params)
        return xgb.XGBRegressor(**model_params)

    def get_hyperparams(self):
        return {
            'max_depth': [1, 2, 3],
            'n_estimators': [25, 50, 75, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 5, 10, 25, 50],
            'colsample_bynode': [0.1, 0.5, 0.9]
        }
    def get_best_hyperparams(self):
        return {'n_estimators': 75, 'min_child_weight': 1,
                'max_depth': 3, 'learning_rate': 0.05,
                'colsample_bynode': 0.5}

class XGBDart(BaseRegressor):
    def make_model(self, params=None):
        model_params = {'objective':'survival:cox', 'tree_method':'hist',
                        'booster':'dart', 'subsample':0.8, 'random_state':0}
        if params:
            model_params.update(params)
        return xgb.XGBRegressor(**model_params)
    def get_hyperparams(self):
        return {
            'max_depth': [1, 2, 3],
            'n_estimators': [25, 50, 75, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 5, 10, 25, 50],
            'colsample_bynode': [0.1, 0.5, 0.9]
        }
    def get_best_hyperparams(self):
        return {'n_estimators': 75, 'min_child_weight': 1,
                'max_depth': 3, 'learning_rate': 0.05,
                'colsample_bynode': 0.5}

class WeibullAFT(BaseRegressor):
    def make_model(self, params=None):
        return sklearn_adapter(WeibullAFTFitter, event_col='Observed')
    def get_hyperparams(self):
        return {
            'alpha': [0.01, 0.05, 0.1, 1]
        }
    def get_best_hyperparams(self):
        return {'alpha': 0.01}