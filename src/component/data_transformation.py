import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer # handling missing values
from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.preprocessing import OrdinalEncoder # ranking data
#pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initialized')

            num_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            cat_cols = ['cut', 'color','clarity']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipline initiated')
            
            num_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipline',num_pipline,num_cols),
                ('cat_pipline',cat_pipline,cat_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_column = ['id', target_column_name]

            input_train_data = train_data.drop(drop_column, axis=1)
            target_train_data = train_data[target_column_name]

            input_test_data = test_data.drop(drop_column, axis=1)
            target_test_data = test_data[target_column_name]

            # transforming using preprocessor
            input_train_data_arr = preprocessor_obj.fit_transform(input_train_data)
            input_test_data_arr = preprocessor_obj.transform(input_test_data)

            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_train_data_arr, np.array(target_train_data)]
            test_arr = np.c_[input_test_data_arr, np.array(target_test_data)]


            save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessor_obj
                )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)