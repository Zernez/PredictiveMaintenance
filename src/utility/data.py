from typing import List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import config as cfg
from ..utility.time import year_week_diff, get_later_week
from dataclasses import dataclass

class Data:

    def __init__(self):
        pass    

    def make_date_dict(self, df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """
        Takes a dataframe and makes a dictionary of all different weeks and an array of all the year-week pairs. Both \
        are sorted.

        :param df: Dataframe of various lengths
        :return: array of dates and corresponding dictionary
        """
        dates = df[['Year', 'Week']].drop_duplicates().to_numpy()
        dates = dates[dates[:, 1].argsort()]
        dates = dates[dates[:, 0].argsort(kind='stable')]
        date_dict = defaultdict()
        for i, week in enumerate(dates):
            date_dict[tuple(week)] = i
        return dates, date_dict

    def make_type_dict(self, types: List[str]) -> dict:
        """
        Makes a dictionary of all the different types of home-care.

        :param types: list of string (names) or equivalent.
        :return: care-type dictionary
        """
        type_dict = defaultdict()
        for i, type_ in enumerate(types):
            type_dict[type_] = i
        return type_dict

    def fix_hc_data_types(self, hc):
        hc['Year'] = pd.Series.astype(hc['Year'], dtype=int)
        hc['Week'] = pd.Series.astype(hc['Week'], dtype=int)
        hc['Minutes'] = pd.Series.astype(hc['Minutes'], dtype=int)
        hc['NumCares'] = pd.Series.astype(hc['NumCares'], dtype=int)
        hc['BirthYear'] = pd.Series.astype(hc['BirthYear'], dtype=int)
        return hc