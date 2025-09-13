import numpy as np
import pandas as pd
from pathlib import Path
import os


class BaseModel:

    def __init__(self):
        self.df = None
        self.read_csv()

    @staticmethod
    def _get_path(name='train_data.csv'):
        full_path = os.path.realpath(__file__)
        p = Path(os.path.dirname(full_path))
        return p.parent / "data" / name


    def read_csv(self):
        p = self._get_path()
        self.df = pd.read_csv(p)

    def preprocess(self):
        pass



