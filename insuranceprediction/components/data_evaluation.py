import csv
from datetime import datetime
import os
import sys
import numpy as np
from insuranceprediction.entity.model_factory import MetricInfoArtifact, ModelFactory

from insuranceprediction.logger import logging
from insuranceprediction.constants import BEST_MODEL_KEY, DS_TARGET_COLUMN_NAME, HISTORY_HEADER, MODEL_PATH_KEY

from insuranceprediction.util.util import load_data, load_data_path, load_numpy_array_data, load_object, read_yaml, write_csv, write_yaml_file

from insuranceprediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from insuranceprediction.entity.config_entity import ModelEvaluationConfig
from insuranceprediction.exception import InsurancePredictionException

class ModelEvaluation:
    def __init__(self,\
                model_evauation_config: ModelEvaluationConfig,
                model_trainer_artifact: ModelTrainerArtifact,
                data_transformation_artifact: DataTransformationArtifact,
                data_ingestion_artifact: DataIngestionArtifact,
                data_validation_artifact: DataValidationArtifact
                ) -> None:
        try:
            self.model_evauation_config = model_evauation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    
    def get_best_model(self) -> object:
        try:
            model = None
            model_evaluation_path = self.model_evauation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_path):
                write_yaml_file(model_evaluation_path, )

                return model

            model_eval_content = read_yaml(file_path=model_evaluation_path)

            model_file_content = dict() if model_eval_content is None else model_eval_content

            if BEST_MODEL_KEY not in model_file_content:
                return model 
            
            model = load_object(file_path=model_eval_content[BEST_MODEL_KEY][MODEL_PATH_KEY])

            return model

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    
    def keep_history(self,model_evaluation_artifact: ModelEvaluationArtifact, previous_best_model:str):
        try:
            history_path = model_evaluation_artifact.model_history_path

            if not os.path.exists(history_path):
                write_csv(history_path,HISTORY_HEADER)
            
            if previous_best_model is not None:
                history_data = [self.model_evauation_config.time_stamp, previous_best_model]
                write_csv(model_evaluation_artifact.model_history_path, history_data)         
                
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    
    def update_evaluation_report(self, model_evaluation_aritifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evauation_config.model_evaluation_file_path
            model_eval_content = read_yaml(eval_file_path)

            model_eval_content = dict() if model_eval_content is None else model_eval_content

            previous_best_model = None

            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Prevoious eval result: {model_eval_content}")

            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_aritifact.evaluated_model_path
                }
            }

            self.keep_history(model_evaluation_artifact=model_evaluation_aritifact, previous_best_model=previous_best_model)
            model_eval_content.update(eval_result)

            logging.info(f"Updated eval result: {model_eval_content}")

            write_yaml_file(file_path=eval_file_path, data= model_eval_content)


        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def initiate_model_evaluation(self):
        try:
            #get the trained model 
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)

            # Get the train and test data 
            model = self.get_best_model()

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data_path(file_path=train_file_path,schema_file_path=schema_file_path)
            test_dataframe = load_data_path(file_path=test_file_path,schema_file_path=schema_file_path)
            schema_content = read_yaml(file_path=schema_file_path)
            target_column_name = schema_content[DS_TARGET_COLUMN_NAME]

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")


            if model is None:
                logging.info(f"Not found any existing model.Hence accepting the trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True, 
                                                                evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                                                                model_history_path=self.model_evauation_config.model_evaluation_history_file_path
                                                                )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact}")
                return model_evaluation_artifact


            model_list = [model,trained_model]

            metric_info_artifact: MetricInfoArtifact = ModelFactory.evaluate_regression_model(model_list=model_list,
                                                                        X_train=train_dataframe,
                                                                        y_train=train_target_arr,
                                                                        X_test=test_dataframe,
                                                                        y_test=test_target_arr,
                                                                        base_accuracy=self.model_trainer_artifact.model_accuracy

                                                                        )

            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True, 
                            evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                            model_history_path=self.model_evauation_config.model_evaluation_history_file_path)
                self.update_evaluation_report(model_evaluation_aritifact=model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact}")
            else:
                logging.info(f"Trained model is not better than the existing model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=False, 
                evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                model_history_path=self.model_evauation_config.model_evaluation_history_file_path
                )

            return model_evaluation_artifact
            
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")