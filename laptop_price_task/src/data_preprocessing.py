import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import json
from bases.base import BaseModel

class LaptopModel(BaseModel):
    folder_name = "laptop_price_task"

    def __init__(self):
        super().__init__()
        self.scores = {}

    def _handle_cpu(self, element):
        # Remove trailing whitespaces, remove clock speed, remove words that don't appear in our db
        element = element.strip().lower()
        element = element.replace("series ", '')
        element = element.replace("quad core ", '')
        element = element.replace("dual core ", '')
        element = element.replace('-', ' ')
        element = " ".join(element.split(" ")[0:-1])
        final = self.scores.get(element, None)
        if final is None:
            keys = self.scores.keys()
            main_series = " ".join(element.split(" ")[:-1])
            for key in keys:
                if main_series in key:
                    final = self.scores[key]
                    break
        return final

    @staticmethod
    def parse_storage_gb_exact(s: str):
        if not isinstance(s, str):
            return pd.Series({"SSD": np.nan, "HDD": np.nan, "FLASH": np.nan})
        totals = {"SSD": 0.0, "HDD": 0.0, "FLASH": 0.0}

        pat = r'(?i)(\d+(?:\.\d+)?)\s*(TB|GB)\s*(SSD|HDD|Flash(?:\s+Storage)?)'
        for size, unit, kind in re.findall(pat, s):
            gb = float(size) * (1024 if unit.upper() == "TB" else 1)
            k = "FLASH" if re.fullmatch(r'(?i)Flash(?:\s+Storage)?', kind) else kind.upper()
            totals[k] += gb
        return pd.Series(totals)

    def preprocess(self):
        scores_path = self._get_path(self.folder_name, name="cpu_score.json")
        with open(scores_path, 'r') as f:
            scores = json.load(f)
            scores = scores['devices']
            for score in scores:
                self.scores[score['name'].lower().replace('-', ' ').strip()] = score['score']


        self.df['touch'] = self.df['ScreenResolution'].str.contains("Touchscreen", case=False, na=False).astype('uint8')
        self.df['px_width'] = self.df['ScreenResolution'].str.split().str[-1].str.split('x').str[0].astype('int32')
        self.df['px_height'] = self.df['ScreenResolution'].str.split().str[-1].str.split('x').str[1].astype('int32')
        self.df['ips_panel'] = self.df["ScreenResolution"].str.contains("IPS", case=True, na=False).astype('uint8')
        self.df['4k'] = self.df['ScreenResolution'].str.contains("4K Ultra").astype('uint8')
        self.df['hd'] = self.df['ScreenResolution'].str.contains(r"HD").astype('uint8')
        self.df['Ram'] = self.df['Ram'].str.replace('GB', '').astype('int32')
        self.df[['SSD', 'HDD', 'FLASH']] = self.df['Memory'].apply(self.parse_storage_gb_exact).astype('float32')
        self.df['Cpu'] = self.df['Cpu'].apply(self._handle_cpu)
        self.df['Weight'] = self.df['Weight'].str.replace('kg', '').astype('float32')
        # oh = pd.get_dummies(self.df["OpSys"], prefix="OpSys", drop_first=True, dtype="uint8")
        # self.df = self.df.drop(columns=["OpSys"]).join(oh)
        self.df = self.df.drop(columns=["OpSys"])
        self.df = self.df.drop(columns=["Memory", "ScreenResolution",
                                        "TypeName", "Company", "Gpu"])
        self.df = self.df.dropna()
        self.standardize()





