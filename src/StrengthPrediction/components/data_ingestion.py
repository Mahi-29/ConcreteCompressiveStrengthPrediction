from src.StrengthPrediction.logger import logging
from src.StrengthPrediction.exception import customexception
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
     