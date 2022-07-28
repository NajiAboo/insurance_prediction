
import sys
import os
import numpy as np
import pandas as pd
from sklearn import pipeline
from yaml import load_all

from insuranceprediction.constants import *
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from insuranceprediction.util.util import read_yaml,load_data, save_numpy_array_data, save_object
from insuranceprediction.logger import logging
from insuranceprediction.exception import InsurancePredictionException
from insuranceprediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from insuranceprediction.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ) -> None:
        try:
            logging.info(f"{'>>' * 30} Data Transformation log started. {'<<' * 30}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise InsurancePredictionException (e,sys) from e

    
    def get_data_transofrmation_object(self) -> ColumnTransformer:
        try:
            schema_config = read_yaml(self.data_validation_artifact.schema_file_path)

            numerical_columns = schema_config[NUMERICAL_COLUMNS]
            categorical_columns = schema_config[CATEGORICAL_COLUMNS]    

            numerical_pipline = pipeline.Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            categorical_pipeline = pipeline.Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ('scalar', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columsn: {categorical_columns}")
            logging.info(f"Numerical colums: {numerical_columns}")

            preprocessing = ColumnTransformer([
                ('num_pipeline', numerical_pipline, numerical_columns),
                ('cat_pipeline', categorical_pipeline, categorical_columns)
            ])

            return preprocessing

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Obtaining preprocessing objec")

            preprocessing_obj = self.get_data_transofrmation_object()

            logging.info(f"Get train and test file")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            schema_info = read_yaml(schema_file_path)

            logging.info(f"Loading the train and test data as pandas framework")
            train_df = load_data(train_file_path, schema_info)
            test_df = load_data(test_file_path, schema_info)

            target_column_name = schema_info[DS_TARGET_COLUMN_NAME]

            logging.info(f"Splitting the train data as features and target column")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
        
            logging.info(f"Splitting the test data as feature and the target column")
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe ")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace('.csv',".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".nps")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info("Savging transformed training and testing array")
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_array)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_array)

            preprocess_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Save preprocess object")
            save_object(preprocess_obj_file_path,preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocess_obj_file_path,
                isTransformed=True,
                message="Successfully transformed the data"
            )

            return data_transformation_artifact

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")


