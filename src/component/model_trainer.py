import sys, os
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerconfig:
    model_obj_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainerconfig = ModelTrainerconfig()

    def initiate_model_trainer(self, train_data,test_data):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            # separating independent and dependent variable
            X_train = train_data[:,:-1]
            X_test = test_data[:,:-1]
            y_train = train_data[:,-1]
            y_test = test_data[:,-1]

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor()
                }
            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)

            logging.info(f'Model Report : {model_report}')

            #to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_object(
                file_path=self.model_trainerconfig.model_obj_path,
                obj=best_model
            )


        except Exception as e:
            raise CustomException(e,sys)

