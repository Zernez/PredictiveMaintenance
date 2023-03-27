from datetime import datetime, timedelta, date
from typing import Union, List, Tuple
import numpy as np
import pandas as pd

class time:

    def __init__(self):
        pass

    def assert_datetime(self, entry: str) -> bool:
        """
        This method checks whether an entry can
        be converted to a valid datatime
        :param entry: entry to check
        :return: boolean value whether conversion successful
        """
        return pd.to_datetime(entry, format='%d-%m-%Y', errors='coerce') is not pd.NaT

    def year_week2date(self, year: Union[int, List[int]], week: Union[int, List[int]]) -> Union[date, List[date]]:
        """
        Takes in a year and a week number and returns a date-variable.

        :param year: int / list
        :param week: int / list
        :return: date variable(s)
        """
        if type(year) == list or type(year) == np.ndarray:
            dates = []
            for i, y in enumerate(year):
                off = 0 if y == 2021 else 1
                dates.append(datetime.strptime(f'{y}-W{week[i] - off}-1', "%Y-W%W-%w").date())
            return dates
        else:
            off = 0 if year == 2021 or year == 2018 else 1
            return datetime.strptime(f'{year}-W{week - off}-4', "%Y-W%W-%w").date()

    def date_after(self, week1: Tuple[int, int], week2: Tuple[int, int]) -> bool:
        """
        Assesses whether the week1 is after week2

        :param week1: (year, week)
        :param week2: (year, week)
        :return: boolean
        """
        diff = self.year_week_diff(week1, week2)
        if diff > 0:
            return True
        return False

    def year_week_diff(self, date1: Tuple[int, int], date2: Tuple[int, int]) -> int:
        """
        Calculates the difference between week1 and week2 in the number of weeks.

        :param date1: (year, week)
        :param date2: (year, week)
        :return: int
        """
        date1 = self.year_week2date(*date1)
        date2 = self.year_week2date(*date2)
        diff = date1 - date2
        return diff.days // 7

    def get_later_week(self, week: Tuple[int, int], week_diff: int) -> Tuple[int, int]:
        """
        Returns a (year, week) object that is week_diff weeks away from date.

        :param week: intitial date (year, week)
        :param week_diff: week difference
        :return: new date
        """
        week = self.year_week2date(*week)
        week = week + timedelta(weeks=week_diff)
        return week.year, week.isocalendar()[1]
