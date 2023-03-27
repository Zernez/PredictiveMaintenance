from operator import index
from pathlib import Path
from typing import List, Any
import pickle
import pandas as pd
import joblib

class FileWriter:

    def __init__(self):
        pass

    def write_km_surv_preds(path: Path, mean, high, low):
        df = pd.concat([mean, high, low], axis=0)
        df.to_csv(path, index=False)

    def write_joblib(path: Path, data):
        joblib.dump(data, path)

    def write_csv(path: Path, df: pd.DataFrame):
        df.to_csv(path, index=False)

    def write_file(*args: Any, path: Path, column_names: List[str] = None):
        """
        Writes a number of lists to a txt-file with the given path

        :param path: The path to where the file is wanted saved
        :param column_names: Names of the coloumns
        :param args: the lists that should be printed of the same length
        :return: None
        """
        num_items = len(args)
        lengths = list(set([len(item) for item in args]))
        max_lengths = [max([len(str(s)) for s in item]) for item in args]
        if len(lengths) > 1:
            raise ValueError("All lists must* have the same length")
        f = open(path, 'w')
        if column_names is not None:
            if len(column_names) != num_items:
                raise ValueError("Number of columns names and lists must be equal")
            for i, name in enumerate(column_names):
                max_lengths[i] = max(max_lengths[i], len(name))
                f.write(("{0: <" + str(max_lengths[i] + 2) + "}").format(name))
            f.write(("\n{0:-<" + str(sum(max_lengths) + 2 * num_items) + "}\n").format(''))
        for i in range(lengths[0]):
            row = ''
            for j in range(num_items):
                row = row + ('{0: <' + str(max_lengths[j] + 2) + '}').format(args[j][i])
            f.write(row + '\n')
        f.close()

    def write_pickle(path: Path, obj: Any):
        """
        Pickles the given object at the given location

        :param path: path to where the object should be pickled
        :param obj: the object to be pickled
        """
        file_handler = open(path, 'wb')
        pickle.dump(obj, file_handler)
        file_handler.close()
        print(f"-- Pickled object at  {path} --")
