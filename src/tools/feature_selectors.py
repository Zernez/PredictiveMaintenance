from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from mrmr import mrmr_regression
from utility.rfe_pi import RFE_PI

class SelectAllFeatures():
    def fit(self, X, y=None):
        self.features = X.columns
        return self

    def get_feature_names_out(self):
        return self.features

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha=0.01)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

class BaseFeatureSelector(ABC):
    """
    Base class for feature selectors.
    """
    def __init__(self, X, y, estimator):
        """Initilizes inputs and targets variables."""
        self.X = X
        self.y = y
        self.estimator = estimator

    @abstractmethod
    def make_model(self):
        """
        """

    def get_features(self) -> List:
        ft_selector = self.make_model()
        ft_selector.fit(self.X, self.y)
        new_features = ft_selector.get_feature_names_out()
        return new_features

class NoneSelector(BaseFeatureSelector):
    def make_model(self):
        return SelectAllFeatures()

class LowVar(BaseFeatureSelector):
    def make_model(self):
        return VarianceThreshold(threshold=0.5)

class SelectKBest10(BaseFeatureSelector):
    def make_model(self):
        return SelectKBest(fit_and_score_features, k=10)

class SelectKBest20(BaseFeatureSelector):
    def make_model(self):
        return SelectKBest(fit_and_score_features, k=20)

class RFE10(BaseFeatureSelector):
    def make_model(self):
        return RFE_PI(self.estimator, n_features_to_select=10, step=0.5)

class RFE20(BaseFeatureSelector):
    def make_model(self):
        return RFE_PI(self.estimator, n_features_to_select=20, step=0.5)

class SFS10(BaseFeatureSelector):
    def make_model(self):
        return SequentialFeatureSelector(self.estimator, n_features_to_select=10,
                                         n_jobs=5,
                                         scoring=fit_and_score_features,
                                         direction="forward")

class SFS20(BaseFeatureSelector):
    def make_model(self):
        return SequentialFeatureSelector(self.estimator, n_features_to_select=20,
                                         n_jobs=-1,
                                         scoring=fit_and_score_features,
                                         direction="forward")

class RegMRMR10(BaseFeatureSelector):
    def make_model(self):
        return mrmr_regression(X=self.X, y=self.y, K=10, show_progress=False)
    def get_features(self):
        return self.make_model()

class RegMRMR20(BaseFeatureSelector):
    def make_model(self):
        return mrmr_regression(X=self.X, y=self.y, K=20, show_progress=False)
    def get_features(self):
        return self.make_model()