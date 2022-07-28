
import logging
import sys
import os

from typing import List
from insuranceprediction.entity.model_factory import GridSearchedBestModel, ModelFactory, MetricInfoArtifact
#from insuranceprediction.entity.model_factory import evaluate_regression_model

from insuranceprediction.util.util import load_numpy_array_data, load_object, save_object

from insuranceprediction.entity.config_entity import DataModelTrainerConfig, DataTransformationConfig
from insuranceprediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from insuranceprediction.exception import InsurancePredictionException
from sklearn.metrics import r2_score,mean_squared_error



class InsuranceEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, model_trainer_config: DataModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact ) -> None:
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Load transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(transformed_train_file_path)       

            logging.info(f"Load transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(transformed_test_file_path)

            logging.info("Splitting the training and test and target features")
            x_train, y_train, x_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]

            logging.info(f"load the model configuration path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Extracting trained model list")
            grid_searched_best_model_list:List[GridSearchedBestModel] = model_factory.grid_searched_best_model_lists(input_feature=x_train,output_feature=y_train)


            model_list = [ model.best_model for model in grid_searched_best_model_list]
            logging.info("Evaluate the modle")

            metric_info: MetricInfoArtifact = ModelFactory.evaluate_regression_model(model_list=model_list,X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,base_accuracy=base_accuracy)
            
            logging.info(f"Best found model on both training and testing dataset.")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object

            logging.info(f"Best found model {model_object}")

            trained_model_file_path = self.model_trainer_config.trained_model_file_path 
            
            insurance_predict_model = InsuranceEstimatorModel(preprocessing_object=preprocessing_obj, trained_model_object=model_object )

            logging.info(f"Saving model at path: {trained_model_file_path}")

            save_object(trained_model_file_path, insurance_predict_model)

            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
                                    trained_model_file_path=trained_model_file_path,
                                    train_rmse=metric_info.train_rmse,
                                    test_rmse=metric_info.test_rmse,
                                    train_accuracy=metric_info.train_accuracy,
                                    test_accuracy=metric_info.test_accuracy,
                                    model_accuracy=metric_info.model_accuracy
            
                                    )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'>>' * 30}Model trainer log completed.{'<<' * 30} ")