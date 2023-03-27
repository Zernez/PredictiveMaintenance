from pathlib import Path
import pandas as pd
import numpy as np
from utility.data import fix_hc_data_types

class RawLoader:
    """
    Class for loading the raw data

    :param path: Path to the folder where the raw data is located
    """

    def __init__(self):
        pass

    def load_assistive_aids(self, filename, path) -> pd.DataFrame:
        """
        This method loads assistive aids data
        :param filename: The name of the file with the data
        :return: A panda dataframe
        """
        converters = {'Personnummer': str, 'Kategori ISO nummer': str}
        df = pd.read_csv(Path.joinpath(path, filename), converters=converters,
                         encoding='iso-8859-10', skiprows=2)
        df = df.replace(r'^\s*$', np.nan, regex=True) # replace empty strs with nan
        df = df.dropna(subset=['Personnummer'])

        # Convert PN to CitizenId
        df['CitizenId'] = df['Personnummer'].str.replace("-", "") \
                            .astype(np.int64) \
                            .apply(lambda x: ((x*8) + 286) * 3) \
                            .astype(str)
        df = df.reset_index(drop=True)

        # Do some renaming
        df = df.rename(columns={'Kategori ISO nummer': 'DevISOClass',
                                'Leveret dato': 'LendDate',
                                'Returneret dato': 'ReturnDate'})
        df = df[['CitizenId', 'DevISOClass', 'LendDate', 'ReturnDate']]

        df['LendDate'] = pd.to_datetime(df['LendDate'], format='%d-%m-%Y')
        df['ReturnDate'] = pd.to_datetime(df['ReturnDate'], format='%d-%m-%Y', errors='coerce')

        return df

    def load_home_care(self, filenames, path) -> pd.DataFrame:
        """
        Parser for the DigiRehab home-care data that holds the number of minutes of home care received for each type
        of care. The parameters are:
         - Year
         - week
         - care type
         - Organisation (Private / municipal)
         - Minutes - Minutes of home care given that week
         - NumCares - how many visits of the given type carried out in the given week
         - sex
         - ID
         - BirthYear

        :param filename: The name of the file with the data
        :return: A panda dataframe
        """
        X = pd.DataFrame()
        for filename in filenames:
            if filename == "Hjemmehjælpdata aug 2022.csv":
                encoding = 'iso-8859-10'
            else:
                encoding = None
            if filename == "Hjemmehjælpdata aug 2022.csv" or filename == 'Hjemmehjælpdata dec 2020.csv':
                converters = {'Personnummer': str}
                df = pd.read_csv(Path.joinpath(path, filename), encoding=encoding,
                                 converters=converters, skiprows=2)
                df = df[df['Personnummer'].str.len() == 11] # remove empty/whitespace strings

                # Convert PN to CitizenId
                df['CitizenId'] = df['Personnummer'].str.replace("-", "").astype(np.int64) \
                                  .apply(lambda x: ((x*8) + 286) * 3).astype(str)

                # Calculate gender and birth year
                df['Gender'] = df['Personnummer'].str.replace("-", "").astype(np.int64) \
                               .apply(lambda x: 'FEMALE' if x % 2 == 0 else 'MALE')
                df['BirthYear'] = df['Personnummer'].str.replace("-", "") \
                                  .str.slice(4,6).astype(int)

                # Do some renaming
                df = df.rename(columns={'År uge': 'Year', 'Ugenummer': 'Week',
                                        'Ydelse navn' : 'CareType',
                                        'Leveret tid (minutter)': 'Minutes',
                                        'Antal ydelser': 'NumCares'})

                # Fix year, convert types
                df['Year'] = [int(x.split('-')[0]) for x in df.Year]
                df = fix_hc_data_types(df)
                df = df[['CitizenId', 'Gender', 'BirthYear', 'Year', 'Week',
                         'Minutes', 'NumCares', 'CareType']]

                X = pd.concat([X, df], ignore_index=True)
            else:
                converters = {'BorgerID': str}
                hc = pd.read_csv(Path.joinpath(path, filename),
                                 converters=converters,
                                 sep=";",
                                 encoding='latin-1')
                hc = hc.dropna(axis=0)
                hc = hc.rename(columns={'År': 'Year', 'Kalender Uge Nr': 'Week',
                                        'Ydelse' : 'CareType', 'Leveret Tid (min)': 'Minutes',
                                        'Antal leverede ydelser': 'NumCares',
                                        'Køn': 'Gender', 'BorgerID': 'CitizenId',
                                        'Født': 'BirthYear'})

                # Fix year, convert types
                hc = fix_hc_data_types(hc)
                hc = hc[['CitizenId', 'Gender', 'BirthYear', 'Year', 'Week',
                         'Minutes', 'NumCares', 'CareType']]

                X = pd.concat([X, hc], ignore_index=True)
        return X