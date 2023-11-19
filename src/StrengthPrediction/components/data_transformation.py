import os
import sys
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

from src.StrengthPrediction.logger import logging
from src.StrengthPrediction.exception import customexception
from src.StrengthPrediction.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    pre_processor_obj_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_preprocessor(self):
        try:
            
            logging.info('initialize the preprocessor')
            columns = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']

            num_pipeline  = Pipeline(

                [
                    ('Missing_value_handler', SimpleImputer()),
                    ("Stander_scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    
                    ('num_pipeline',num_pipeline,columns)
                ]
            )
            logging.info(
                "Successfully created the preprocessor"
                         )
            return preprocessor



        except Exception as e:
            raise customexception(e, sys)
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Data imported successfully")
            logging.info(f"Training dataset head: \n {train_data.head().to_string()}")
            logging.info(f"Test dataset head: \n {test_data.head().to_string()}")

            preprocessor=self.get_data_preprocessor()
            logging.info('preprocessor object loaded successfully')
            dependant_feature = 'concrete_compressive_strength'
            drop_columns = [dependant_feature]

            input_feature_training_df = train_data.drop(drop_columns,axis=1)
            output_feature_training_df = train_data[dependant_feature]
            input_feature_test_df = test_data.drop(drop_columns,axis=1)
            output_feature_test_df = test_data[dependant_feature]
            
            logging.info("data split in dependent and independent feature completed")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_training_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Data transformation completed successfully")

            train_arr = np.c_[input_feature_train_arr,np.array(output_feature_training_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(output_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.pre_processor_obj_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr
            )





        except Exception as e:
            raise customexception(e, sys)
        
        
    