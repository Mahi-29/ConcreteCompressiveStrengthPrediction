import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
from pathlib import Path

from src.StrengthPrediction.logger import logging
from src.StrengthPrediction.exception import customexception

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

 
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv("notebooks/data/concrete_data.csv")

            logging.info("Data loading successful")

            data.drop_duplicates(inplace=True)

            logging.info("Duplicates Dropped")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw data store successfully")

            train_data,test_data = train_test_split(data, test_size=0.25)
            logging.info("performed train test split")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Train and test csv files are stored")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )




        except Exception as e:
            logging.info(f"{e} Error occurred")
            raise customexception(e,sys)
        