import os
import sys

from src.StrengthPrediction.components.data_ingestion import DataIngestion
from src.StrengthPrediction.components.data_transformation import DataTransformation
from src.StrengthPrediction.components.model_trainer import ModelTrainer
from src.StrengthPrediction.components.model_evaluation import ModelEvaluation

obj = DataIngestion()

train_path, test_path = obj.initiate_data_ingestion()

datatransform = DataTransformation()

train_arr, test_arr = datatransform.initialize_data_transformation(train_path, test_path)

trainer = ModelTrainer()

trainer.initiate_model_training(train_arr, test_arr)

evaluator = ModelEvaluation()

evaluator.evaluate(test_arr)

