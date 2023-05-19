import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from lifelines import WeibullAFTFitter
from mrmr import mrmr_regression
import umap
#from utility.rfe_pi import RFE_PI

class SelectAllFeatures():
    def fit(self, X, y=None):
        self.features = X.columns
        return self

    def get_feature_names_out(self):
        return self.features

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha=0.1)
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
        if ft_selector.__class__.__name__ == "UMAP":
            self.fit(ft_selector, self.X)
            new_features = self.get_feature_names_out()
        else:
            ft_selector.fit(self.X, self.y)
            new_features = ft_selector.get_feature_names_out()
        return new_features

class NoneSelector(BaseFeatureSelector):
    def make_model(self):
        return SelectAllFeatures()

class LowVar(BaseFeatureSelector):
    def make_model(self):
        return VarianceThreshold(threshold=0.1) #0.5

class SelectKBest4(BaseFeatureSelector):
    def make_model(self):
        return SelectKBest(fit_and_score_features, k= 4)

class SelectKBest8(BaseFeatureSelector):
    def make_model(self):
        return SelectKBest(fit_and_score_features, k= 8)

# class RFE4(BaseFeatureSelector):
#     def make_model(self):
#         return RFE_PI(self.estimator, n_features_to_select= 4, step=0.5)

# class RFE8(BaseFeatureSelector):
#      def make_model(self):
#          return RFE_PI(self.estimator, n_features_to_select= 8, step=0.5)

class SFS4(BaseFeatureSelector):
    def make_model(self):
        return SequentialFeatureSelector(self.estimator, n_features_to_select= 4,
                                         scoring=fit_and_score_features,
                                         direction="forward")

class SFS8(BaseFeatureSelector):
    def make_model(self):
        return SequentialFeatureSelector(self.estimator, n_features_to_select= 8,
                                         scoring=fit_and_score_features,
                                         direction="forward")

class RegMRMR4(BaseFeatureSelector):
    def make_model(self):
        return mrmr_regression(X=self.X, y=self.y, K=4, show_progress=False)
    def get_features(self):
        return self.make_model()

class RegMRMR8(BaseFeatureSelector):
     def make_model(self):
         return mrmr_regression(X=self.X, y=self.y, K=8, show_progress=False)
     def get_features(self):
         return self.make_model()
    
class UMAP8(BaseFeatureSelector):
    def make_model(self):
        self.components= 8
        return umap.UMAP(n_components= self.components)
    
    def fit(self, ft_selector, X, y= None):
        X = ft_selector.fit_transform(X)

        labels= []
        for element in range (1, self.components + 1 ,1):
            labels.append("UMAP_Feature_"+ str(element))

        self.features = pd.DataFrame(X, columns = labels)      
        return self

    def get_feature_names_out(self):
        return self.features

class UMAP12(BaseFeatureSelector):
    def make_model(self):
        self.components= 8
        return umap.UMAP(n_components= self.components)
    
    def fit(self, ft_selector, X, y= None):
        X = ft_selector.fit_transform(X)

        labels= []
        for element in range (1, self.components + 1 ,1):
            labels.append("UMAP_Feature_"+ str(element))

        self.features = pd.DataFrame(X, columns = labels)      
        return self

    def get_feature_names_out(self):
        return self.features
