from xmlrpc.client import boolean
import numpy as np
from typing import Tuple, Union
from utility.data import Citizen
from utility.time import year_week_diff, get_later_week
import config as cfg
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, boxcox

class Preprocessor:

    def series_to_moving_average(df_ts: pd.DataFrame,
                                window_len: int,
                                lag: int,
                                lbl_cols: List) -> pd.DataFrame:
        cols = list(df_ts.columns.drop(lbl_cols))
        total_df = pd.DataFrame()
        for period in df_ts['Period'].unique():
            period_df = df_ts.loc[df_ts['Period'] == period].copy(deep=True) # Get period data
            for ft_col in cols:
                roll = period_df.groupby(['Id', 'Period'])[ft_col].rolling(window_len)
                ma = roll.mean().shift(lag).reset_index(0, drop=True)[period]
                period_df[ft_col] = ma
            period_df = period_df.dropna().reset_index(drop=True)
            total_df = pd.concat([total_df, period_df], axis=0)
        return total_df