import pandas as pd
import numpy as np

class Cleaner:

    def __init__(self):
        pass

    def clean_ats(self, df: pd.DataFrame, iso_classes: pd.DataFrame) -> pd.DataFrame:
        # Remove invalid values
        df = df.sort_values(['CitizenId', 'LendDate'])
        df = df[df['CitizenId'] != "0000000000"]
        df = df[df['CitizenId'] != '0']
        df = df[df['CitizenId'] != "#VALUE!"]
        df = df[df['CitizenId'] != '681']
        df = df.dropna(subset=['CitizenId'])

        # Only include known ats, shorten iso and fix dates
        df = df[df['DevISOClass'].isin(iso_classes['DevISOClass'])] # Only include known ISO types
        df['DevISOClass'] = df['DevISOClass'].apply(lambda x: x[:6]) # Shorten ISO class
        df = df.fillna(df.LendDate.max()) # Replace invalid return dates with latest obs lend date
        df = df.loc[df['ReturnDate'] >= df['LendDate']] # Return date must be same or later than lend date
        return df

    def clean_home_care(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans special danish characters

        :param df: the data frame that has the 'CareType' parameter
        :return: clean dataframe
        """
        # Cleans special danish characters and redundant information
        df['CareType'] = df['CareType'].map(self.clean_string)

        # Use only received home care in minutes
        df = df.loc[df['Minutes'] > 0]

        return df

    def clean_string(self, string: str) -> str:
        string = string.replace('Ã¸', 'ø')
        string = string.replace('Ã¦', 'æ')
        string = string.replace('Ã¥', 'å')
        string = string.replace(' (FSIII)', '')
        return string