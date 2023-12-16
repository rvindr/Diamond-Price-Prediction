import os, sys
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer


if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initialize_data_ingestion()
    print(train_data_path,test_data_path)

    data_transformation =  DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer_obj = ModelTrainer()
    model_trainer_obj.initiate_model_trainer(train_arr,test_arr)
    print('all ok')
