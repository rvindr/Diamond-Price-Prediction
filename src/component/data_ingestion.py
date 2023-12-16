import sys
import os
import pandas as pd 
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
# data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    row_data_path = os.path.join('artifacts','row.csv')


# initialize data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()


    def initialize_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv('notebook/data/gemstone.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path, index=False)

            logging.info("Raw data is created")

            train_set,test_set = train_test_split(df, test_size=0.30, random_state=30)

            train_set.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, header=True, index=False)

            logging.info('Data Ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )




        except Exception as e:
            logging.info('Error occured in data ingestion method')
            raise CustomException(e,sys)