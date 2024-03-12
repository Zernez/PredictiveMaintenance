from abc import ABC, abstractmethod
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import config as cfg
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sksurv.svm import FastSurvivalSVM
from auton_survival import DeepCoxPH
from auton_survival import DeepSurvivalMachines
from bnnsurv import models
import numpy as np

class BaseRegressor (ABC):
    """
    Base class for regressors.
    """
    def __init__ (self):
        """Initilizes inputs and targets variables."""

    @abstractmethod
    def make_model (self, params=None):
        """
        This method is an abstract method to be implemented
        by a concrete classifier. Must return a sklearn-compatible
        estimator object implementing 'fit'.
        """

    @abstractmethod
    def get_hyperparams (self):
        """Method"""

    @abstractmethod
    def get_best_hyperparams (self):
        """Method"""

    def get_tuneable_params (self):
        return self.get_hyperparams()

    def get_best_params (self):
        return self.get_best_hyperparams()

    def get_estimator (self, params=None):
        return self.make_model(params)
    
class CoxPH(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        return {'n_iter': [10, 50, 100],
                'tol': [1e-1, 1e-5, 1e-9]}
    
    def get_best_hyperparams (self):
        return {'tol': 1e-3,
                'n_iter': 20}

class RSF(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_RSF
        if params:
            model_params.update(params)
        return RandomSurvivalForest(**model_params)
    
    def get_hyperparams (self):
        return {'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 400],
                'min_samples_split': [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)],
                'min_samples_leaf': [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]}
    
    def get_best_hyperparams (self):
        return  {'n_estimators': 3,
                 'min_samples_split': 15,
                 'min_samples_leaf': 600, 
                 'max_depth': 5}
    
class DeepSurv(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DEEPSURV
        if params:
            model_params.update(params)
        return DeepCoxPH(layers=model_params['layers'])
    
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'iters': [100, 300, 500, 1000],
                'layers': [[16], [32], [64]]
                }
    
    def get_best_hyperparams(self):
        return {'batch_size' : 32,
                'learning_rate' : 1e-2,
                'iters': 300,
                'layers': [32]}
    
class DSM(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DSM
        if params:
            model_params.update(params)
        return DeepSurvivalMachines(layers=model_params['layers'])
    
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'iters': [100, 300, 500, 1000],
                'layers': [[16], [32], [64]]}
            
    def get_best_hyperparams(self):
        return {'batch_size' : 32,
                'learning_rate' : 1e-3,
                'iters': 100,
                'layers': [32]}
    
class BNNmcd(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_BNN
        if params:
            model_params.update(params)
        return models.MCD(**model_params)
    
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'num_epochs': [10, 25, 50],
                'layers': [[16], [32], [64]]
                }

    def get_best_hyperparams(self):
        return {'batch_size' : 32,
                'learning_rate' : 1e-2,
                'num_epochs': 10,
                'layers': [32]}