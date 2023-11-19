import os
import sys

from src.StrengthPrediction.components.data_ingestion import DataIngestion



obj = DataIngestion()

train_path, test_path = obj.initiate_data_ingestion()
