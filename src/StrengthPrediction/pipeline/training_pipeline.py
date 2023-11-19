import os
import sys

from src.StrengthPrediction.components.data_ingestion import DataIngestion
from src.StrengthPrediction.components.data_transformation import DataTransformation


obj = DataIngestion()

train_path, test_path = obj.initiate_data_ingestion()

datatransform = DataTransformation()

train_arr, test_arr = datatransform.initialize_data_transformation(train_path, test_path)

print(test_arr)

