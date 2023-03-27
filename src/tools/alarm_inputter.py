import pandas as pd
import numpy as np

class AlarmInputter:

    def __init__(self):
        pass

    def make_hc_features(self, df: pd.DataFrame, citizen_idx_map: dict,
                        dates: np.ndarray, date_dict: dict, care_dict: dict) -> np.ndarray:
        """
        Encodes all observations for a single person to a sequence of variables containing
        the number of minutes of care received of that type.
        :param df: All observations for a single person (ID)
        :param citizen_idx_map: a mapping from citizen id to index
        :param care_dict: the care dictionary
        :return: An array where the two first columns are the year and week
        and the rest correspond to each type of care elements are number of minutes
        of care received in that week of that type of care.
        """
        minutes_care = np.zeros((len(citizen_idx_map), len(dates), len(care_dict)), np.float32)
        home_care = list(df[['CitizenId', 'Year', 'Week', 'CareType', 'Minutes', 'NumCares']].to_records(index=False))
        for obs in home_care:
            minutes_care[citizen_idx_map[obs[0]], date_dict[(obs[1], obs[2])],
                        care_dict[obs[3]]] += obs[4]*obs[5]
        return np.around(minutes_care)