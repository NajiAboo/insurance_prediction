

import importlib
import logging
import sys
import numpy as np
import pandas as pd
from typing import List
from collections import namedtuple
from insuranceprediction.util.util import read_yaml
from insuranceprediction.exception import InsurancePredictionException
from sklearn.metrics import r2_score,mean_squared_error

GRID_SEARCH = "grid_search"
GRID_SEARCH_MODULE = "module"
GRID_SEARCH_CLASS = "class"
GRID_SEARCH_PARAMS = "params"
MODEL_SELECTION = "model_selection"
MODULE_KEY = "module"
CLASS_NAME_KEY ="class"
PARAM_KEY = "params"
SEARCH_PARAM_GRID_KEY ="search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score"])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

class ModelFactory:
    def __init__(self, model_config_path:str) -> None:
        try:
            self.model_config_path:str = model_config_path
            self.config:dict = read_yaml(model_config_path)

            self.grid_search_cv_module:str = self.config[GRID_SEARCH][GRID_SEARCH_MODULE]
            self.grid_search_class_name:str = self.config[GRID_SEARCH][GRID_SEARCH_CLASS]
            self.grid_search_property_data:dict = dict(self.config[GRID_SEARCH][GRID_SEARCH_PARAMS])

            self.model_initialization_config:dict = self.config[MODEL_SELECTION]

            self.initialized_model_list = self.initialize_model_list()
            
            self.grid_searched_best_model_list = []

        except Exception as e:
            raise InsurancePredictionException(e,sys)

    def grid_searched_best_model_lists(self, input_feature, output_feature) -> List[GridSearchedBestModel]:
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module, class_name=self.grid_search_class_name)

            for initialized_model in self.initialize_model_list:

                grid_search_best_model = self.execute_grid_search_operations(initialized_model=initialized_model, input_feature=input_feature,output_feature=output_feature)
                self.grid_searched_best_model_list.append(grid_search_best_model)

            return self.grid_searched_best_model_list

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def execute_grid_search_operations(self, initialized_model: InitializedModelDetail, 
                                      input_feature, output_feature):
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(
                                                                module_name=self.grid_search_cv_module,
                                                                class_name=self.grid_search_class_name
                                                            )
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)

            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,self.grid_search_property_data)

            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)

            grid_search_cv.fit(input_feature,output_feature)
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__}" completed {"<<"*30}'
            logging.info(message)

            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            
            return grid_searched_best_model

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e


    def initialize_model_list(self) -> List[InitializedModelDetail]:
        try:
            logging.info(f"Started model list initialization")
            logging.info(f"initialized model list config : {self.model_initialization_config}")
            initialized_model_list = []

            for model_serial_number in self.model_initialization_config.keys():
                model_initialized_config = self.model_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialized_config[MODULE_KEY], class_name=model_initialized_config[CLASS_NAME_KEY])
                
                model = model_obj_ref()

                if PARAM_KEY in model_initialized_config:
                    model_params = dict(model_initialized_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model, property_data=model_params)

                param_grid_search = model_initialized_config[SEARCH_PARAM_GRID_KEY]

                model_name = f"{model_initialized_config[MODULE_KEY]}.{model_initialized_config[CLASS_NAME_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialize_model_list = initialized_model_list
            logging.info(f"Initialized model list : {initialized_model_list}")
            return self.initialize_model_list

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    @staticmethod
    def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
        try:
            
        
            index_number = 0
            metric_info_artifact = None
            for model in model_list:
                model_name = str(model)  #getting model name based on model object
                logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
                
                #Getting prediction for training and testing dataset
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                #Calculating r squared score on training and testing dataset
                train_acc = r2_score(y_train, y_train_pred)
                test_acc = r2_score(y_test, y_test_pred)
                
                #Calculating mean squared error on training and testing dataset
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                # Calculating harmonic mean of train_accuracy and test_accuracy
                model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
                diff_test_train_acc = abs(test_acc - train_acc)
                
                #logging all important metric
                logging.info(f"{'>>'*30} Score {'<<'*30}")
                logging.info(f"Train Score\t\t Test Score\t\t Average Score")
                logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

                logging.info(f"{'>>'*30} Loss {'<<'*30}")
                logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
                logging.info(f"Train root mean squared error: [{train_rmse}].")
                logging.info(f"Test root mean squared error: [{test_rmse}].")


                #if model accuracy is greater than base accuracy and train and test score is within certain thershold
                #we will accept that model as accepted model
                if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                            model_object=model,
                                                            train_rmse=train_rmse,
                                                            test_rmse=test_rmse,
                                                            train_accuracy=train_acc,
                                                            test_accuracy=test_acc,
                                                            model_accuracy=model_accuracy,
                                                            index_number=index_number)

                    logging.info(f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1
            if metric_info_artifact is None:
                logging.info(f"No model found with higher accuracy than base accuracy")
            return metric_info_artifact
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        try:
            module_name = importlib.import_module(module_name)
            logging.info(f"module name : {module_name}")
            class_name_ref = getattr(module_name, class_name)
            return class_name_ref
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("parameter propertly should be dictionary")
            for key,value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e