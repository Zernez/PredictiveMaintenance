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

class WeibullAFT (BaseRegressor):
    def make_model (self, params=None):
        return sklearn_adapter (WeibullAFTFitter, event_col='Event')
    
    def get_hyperparams (self):
        return {'alpha': [0.03, 0.05, 0.07],
                'penalizer': [0.06, 0.07, 0.08]}
    
    def get_best_hyperparams (self):
        return {'alpha': 0.05,
                'penalizer': 0.07}
 
class LogNormalAFT (BaseRegressor):
    def make_model (self, params=None):
        return sklearn_adapter (LogNormalAFTFitter, event_col='Event')
    
    def get_hyperparams(self):
        if cfg.DATASET_NAME == "xjtu":
            return {'alpha': [0.03, 0.05, 0.07],
                    'penalizer': [0.06, 0.07, 0.08]}
        elif cfg.DATASET_NAME == "pronostia":
            return {'alpha': [0.03, 0.05, 0.07],
                    'penalizer': [0.17, 0.18, 0.19]}

    
    def get_best_hyperparams(self):
        return {'alpha': 0.05,
                'penalizer': 0.07}
    
class LogLogisticAFT (BaseRegressor):
    def make_model (self, params=None):
        return sklearn_adapter (LogLogisticAFTFitter, event_col='Event')
    
    def get_hyperparams (self):
        return {'alpha': [0.03, 0.05, 0.1]}
    
    def get_best_hyperparams (self):
        return {'alpha': 0.03}
    
class CoxPH (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'n_iter': [5],
                    'tol': [1e-1]} 
        else:
            return {'n_iter': [10, 15, 20],
                    'tol': [1e-1, 1e-2]}
    
    def get_best_hyperparams (self):
        return {'tol': 1e-3,
                'n_iter': 20}

class CphRidge (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH_RIDGE
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'n_iter': [5],
                    'tol': [1e-1]} 
        else:
            return {'n_iter': [10, 15, 20],
                    'tol': [1e-1, 1e-2]}
    
    def get_best_hyperparams (self):
        return {'tol': 1e-3,
                'n_iter': 20}

class CphLASSO (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH_LASSO
        if params:
            model_params.update(params)
        return CoxnetSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        return {'n_alphas': [50, 100, 200],
                'normalize': [True, False],
                'tol': [1e-1, 1e-3, 1e-5],
                'max_iter': [100]}
    
    def get_best_hyperparams (self):
        return {'tol': 1e-03,
                'normalize': True,
                'n_alphas': 50,
                'max_iter': 100}

class CphElastic (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH_ELASTIC
        if params:
            model_params.update(params)
        return CoxnetSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        return {'n_alphas': [50, 100, 200],
                'normalize': [True, False],
                'tol': [1e-1, 1e-3, 1e-5],
                'max_iter': [100]}
    
    def get_best_hyperparams (self):
        return {'tol': 1e-03,
                'normalize': True,
                'n_alphas': 50,
                'max_iter': 100}

class RSF (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_RSF
        if params:
            model_params.update(params)
        return RandomSurvivalForest(**model_params)
    
    def get_hyperparams (self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'max_depth': [3, 4, 5],
                    'n_estimators': [15, 30, 45],
                    'min_samples_split': [2, 4, 8],
                    'min_samples_leaf': [1, 2, 3]}
        else:
            return {'max_depth': [3, 5, 7],
                    'n_estimators': [25, 50, 75],
                    'min_samples_split': [2, 5, 8],
                    'min_samples_leaf': [1, 2, 3]}      
    
    def get_best_hyperparams (self):
        return  {'n_estimators': 3, 
                 'min_samples_split': 15,
                 'min_samples_leaf': 600, 
                 'max_depth': 5}
    
class CoxBoost (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_GRADBOOST
        if params:
            model_params.update(params)
        return GradientBoostingSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'learning_rate': [0.1, 0.08, 0.05],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 4, 8],
                    'min_samples_leaf': [1, 2, 3],
                    'tol': [1e-1]}
        else:
            return {'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'tol': [1e-1, 1e-2]}
    
    def get_best_hyperparams (self):
        return  {'learning_rate': 0.1,
                 'n_estimators' : 110, 
                 'loss': 'coxph', 
                 'dropout_rate': 0.0,
                 'max_depth': 3,
                 'min_samples_split': 3,
                 'min_samples_leaf': 2}

class GradientBoostingDART (BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_GRADBOOST_DART
        if params:
            model_params.update(params)
        return GradientBoostingSurvivalAnalysis(**model_params)
    
    def get_hyperparams (self):
        return {'learning_rate': [0.1, 0.05, 0.01],
                'n_estimators': [100 ,110 , 120],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 2],
                'dropout_rate': [0.2, 0.1],
                'tol': [1e-2, 1e-3, 1e-4]}
    
    def get_best_hyperparams (self):
        return  {'learning_rate': 0.1, 
                'n_estimators': 110,
                 'dropout_rate': 0.1,
                 'max_depth': 4,
                 'min_samples_split': 3,
                 'min_samples_leaf': 1}
    
class SVM (BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_SVM
        if params:
            model_params.update(params)
        return FastSurvivalSVM(**model_params)
    
    def get_hyperparams (self):
        return {'alpha': [0, 1, 2, 3],
                'rank_ratio': [0.4, 0.5, 0.6, 0.7, 0.8],
                'max_iter': [80, 100, 120],
                'optimizer': ['alvtree', 'direct-count', 'PRSVM', 'rbtree']}
    
    def get_best_hyperparams (self):
        return  {'alpha': 1, 
                 'rank_ratio': 0.8, 
                 'max_iter': 120, 
                 'optimizer': 'direct-count'}
    
class DeepSurv(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DEEPSURV
        if params:
            model_params.update(params)
        return DeepCoxPH(layers=model_params['layers'])
    
    def get_hyperparams(self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'batch_size' : [4],
                    'learning_rate' : [1e-1, 1e-2],
                    'iters': [50, 100],
                    'layers': [[16]]
                    }
        else:
            return {'batch_size' : [16, 32],
                    'learning_rate' : [1e-1, 1e-2, 1e-3],
                    'iters': [100, 300, 500],
                    'layers': [[16], [32]]
                    }
    
    def get_best_hyperparams(self):
        return {'batch_size' : 10,
                'learning_rate' : 1e-4,
                'iters': 100,
                'layers': [32]}
    
class DSM(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DSM
        if params:
            model_params.update(params)
        return DeepSurvivalMachines(layers=model_params['layers'])
    
    def get_hyperparams(self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'batch_size' : [4],
                    'learning_rate' : [1e-1, 1e-2],
                    'iters': [50, 100],
                    'layers': [[16]]
                    }
        else:
            return {'batch_size' : [16, 32],
                    'learning_rate' : [1e-1, 1e-2, 1e-3],
                    'iters': [100, 300, 500],
                    'layers': [[16], [32]]
                    }
            
    def get_best_hyperparams(self):
        return {'batch_size' : 32,
                'learning_rate' : 1e-4,
                'iters': 100,
                'layers': [32]}
    
class BNNmcd(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_BNN
        if params:
            model_params.update(params)
        return models.MCD(**model_params)
    
    def get_hyperparams(self):
        if cfg.DATA_TYPE == "bootstrap":
            return {'batch_size' : [4],
                    'learning_rate' : [1e-1, 1e-2],
                    'num_epochs': [5],
                    'layers': [[16]]
                    }
        else:
            return {'batch_size' : [16, 32],
                    'learning_rate' : [1e-2, 1e-3],
                    'num_epochs': [5, 10],
                    'layers': [[16], [32]]
                    }

    def get_best_hyperparams(self):
        return {'batch_size' : 10,
                'learning_rate' : 1e-4,
                'num_epochs': 10,
                'layers': [32]}
