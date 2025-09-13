import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from laptop_price_task.base import BaseModel

class LaptopModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.scores = {}

    def _handle_cpu(self, element):
        # Remove trailing whitespaces, remove clock speed, remove words that don't appear in our db
        element = element.strip()
        element = element.replace("series ", '')
        element = element.replace("quad core ", '')
        element = element.replace("dual core ", '')
        element = element.replace('-', ' ')
        element = " ".join(element.split(" ")[0:-1])
        return self.scores.get(element, None)



    def preprocess(self):
        scores_path = self._get_path(name="cpu_scores.json")
        with open(scores_path, 'r') as f:
            scores = json.load(f)
            scores = scores['devices']
            for score in scores:
                self.scores[score['name'].lower().replace('-', ' ').strip()] = score['score']


        self.df['touch'] =
        self.df['px_width'] = 0
        self.df['px_height'] = 0
        self.df['ips_panel'] =