import pickle
from typing import Any, List
from pathlib import Path
import pandas as pd
import joblib

class FileReader:

    def __init__(self):
        pass

    def read_km_surv_preds(path: Path):
        df = pd.read_csv(path)
        mean = pd.DataFrame(df.iloc[0]).T
        high = pd.DataFrame(df.iloc[1]).T
        low = pd.DataFrame(df.iloc[2]).T
        return mean, high, low

    def read_joblib(path: Path):
        return joblib.load(path)

    def read_csv(path: Path,header: str='infer',
                sep: str=',', usecols: List[int]=None,
                names: List[str]=None, converters: dict=None,
                encoding=None, skiprows=None, parse_dates=None) -> pd.DataFrame:
        return pd.read_csv(path, header=header, sep=sep, usecols=usecols,
                        names=names, converters=converters,
                        encoding=encoding, skiprows=skiprows,
                        parse_dates=parse_dates)

    def read_pickle(path: Path) -> Any:
        """
        Loads the pickled object at the location given.

        :param path: Path (including the file itself)
        :return: obj
        """
        file_handler = open(path, 'rb')
        obj = pickle.load(file_handler)
        file_handler.close()
        return obj