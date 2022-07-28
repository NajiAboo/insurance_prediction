
import os
import sys
import json
from termios import ECHOE
from typing import Tuple
from evidently import dashboard
import pandas as pd
from insuranceprediction.components import data_ingestion
from insuranceprediction.constants import *

from insuranceprediction.logger import logging
from insuranceprediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from insuranceprediction.entity.config_entity import DataValidationConfig
from insuranceprediction.exception import InsurancePredictionException
from insuranceprediction.util.util import read_json, read_yaml

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact) -> None:
        try:
            logging.info(f"{'>>'*30} Data validation log started. {'<<'*30}\n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e
    
    def is_train_test_file_exist(self)->bool:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_exist = os.path.exists(train_file_path) 
            is_test_exist = os.path.exists(test_file_path)

            is_train_test_exist = is_train_exist and is_test_exist

            logging.info(f"Is train and test file exist? {is_train_test_exist}")

            if not is_train_test_exist:
                message = f"Training file: {train_file_path} or Testing file:  {test_file_path} is not present"
                raise Exception(message)

            return is_train_test_exist

        except Exception as e:
            raise InsurancePredictionException(e, sys)

    def is_number_of_columns_matched(self,ds_column_info:dict,df:pd.DataFrame, ds_file_path:str)-> bool:
        try:
             
            config_column_count = len(ds_column_info)
            df_columns_count = len(df.columns)

            is_column_count_matched = config_column_count == df_columns_count

            logging.info(f"Column match done. column match with {ds_file_path} status is: {is_column_count_matched}")

            if not is_column_count_matched:
                message = f""" Column mismatch found with {ds_file_path}.
                schema has {config_column_count} number of columns and 
                dataset file has got {df_columns_count}
                """
                raise Exception(message)

            return is_column_count_matched


        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def is_columns_names_matched(self,ds_column_info:dict,ds:pd.DataFrame, ds_file_path:str)-> bool:
        try:
            column_name_match = False
            config_column_names = list(ds_column_info)

            dataset_column_names = list(ds.columns)

            column_name_match = all(cnf==dsf for cnf, dsf in zip(config_column_names, dataset_column_names))

            logging.info(f"Columns match completed bewteen config columns and {ds_file_path}. Status found : {column_name_match}")

            if not column_name_match:
                message = f""" Column name mismatch found between config columns and {ds_file_path}.
                config conlumns : {config_column_names} and dataset columns : {dataset_column_names}
                """
                raise Exception(message)
            

            return column_name_match
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e
    
    def is_categorical_matched(self,config_cat_values:list, dataset_cat_values:list, cat_feature_name:str) -> bool:
        
        try:
            is_matched = False

            is_matched = all(cat in config_cat_values for cat in dataset_cat_values)

            if not is_matched:
                message = f""" Categorical values are not matching for the feature name : {cat_feature_name}.
                Configuration categorical values for {cat_feature_name}: {config_cat_values}. 
                dataset categorical values for {cat_feature_name}: {dataset_cat_values}
                """
                raise Exception(message)

            return is_matched
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def is_categorical_values_matched(self, config_contegorical:dict,input_dataset:pd.DataFrame) -> bool:
        try:
            is_matched = True
            config_categorical_features = config_contegorical[DS_CATEGORICAL_COLUMNS]

            for cat_feature in config_categorical_features:
                ds_categorical_values = list(input_dataset[cat_feature].unique())
                config_categorical_values = config_contegorical[DS_DOMAIN_VALUES][cat_feature]
                is_matched &= self.is_categorical_matched(config_categorical_values, ds_categorical_values, cat_feature)
            
            if not is_matched:
                message= f"""
                    Categorical features are not matched. {config_categorical_features}
                """
                raise Exception(message)

            return is_matched
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e

    def validate_dataset_schema(self) -> bool:
        try:
            is_data_validated = True

            schema_info = read_yaml(self.data_validation_config.schema_file_path)
            schema_columns_info = schema_info[DS_COLUMNS_INFO]

            train_ds = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_ds = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #validate train dataset
            is_data_validated &= self.is_number_of_columns_matched(schema_columns_info,train_ds,self.data_ingestion_artifact.train_file_path)
            #validate test dataset
            is_data_validated &= self.is_number_of_columns_matched(schema_columns_info,test_ds,self.data_ingestion_artifact.test_file_path)
            # train dataset column matching
            is_data_validated &= self.is_columns_names_matched(schema_columns_info,train_ds, self.data_ingestion_artifact.train_file_path)
            # test dataset column matching
            is_data_validated &= self.is_columns_names_matched(schema_columns_info,test_ds, self.data_ingestion_artifact.test_file_path)

            # Check categorical values matching
            is_data_validated &= self.is_categorical_values_matched(schema_info,train_ds)
            is_data_validated &= self.is_categorical_values_matched(schema_info, test_ds)

            if not is_data_validated:
                message = f"""
                Something went wrong wit the validation, it should be shown in the previous functions.
                """
                raise Exception(message)
            
            return is_data_validated

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def get_last_pushed_raw_path(self) -> str:
        try:
            last_pushed_config_path = os.path.join(
                ROOT_DIR,
                CONFIG_DIR,
                LAST_PUSHED_MODEL_CONFIG_INFO
            )

            pushed_config = read_yaml(last_pushed_config_path)
            last_pushed_raw_path = pushed_config[PUSHED_MODEL_INGESTION_RAW_PATH]
            
            return last_pushed_raw_path
        except Exception as ex:
            raise  InsurancePredictionException(ex,sys) from ex

    
    def save_data_drift_report(self, source_path:str, target_path:str, report_file_path:str):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            source_df = pd.read_csv(source_path)
            target_path = pd.read_csv(target_path)
            
            profile.calculate(source_df,target_path)
            
            report = json.loads(profile.json())

            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            
            return report

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
    
    def save_data_drift_report_page(self, source_path:str, target_path:str, report_file_path:str):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])

            source_df = pd.read_csv(source_path)
            target_path = pd.read_csv(target_path)
            
            dashboard.calculate(source_df,target_path)
            
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            dashboard.save(report_file_path)

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def get_and_save_data_drift_reports(self):
        try:
            last_pushed_raw_path = self.get_last_pushed_raw_path()

            is_last_pushed_exist = len(last_pushed_raw_path) < 1 and \
                                   os.path.exists(last_pushed_raw_path)  

            if not is_last_pushed_exist:
                message = f""" Last pushed train and test is not found. 
                Data drift check done on the train and test data
                """
                logging.info(message)

            if is_last_pushed_exist:
                raw_data_file_path = self.data_ingestion_artifact.raw_file_path
                # Compare last pushed raw data with current data
                self.save_data_drift_report(raw_data_file_path, 
                                            last_pushed_raw_path,
                                            self.data_validation_config.pushed_report_file_path
                                            )
            
            # Compare the train and test data drift
            self.save_data_drift_report(self.data_ingestion_artifact.train_file_path,
                                        self.data_ingestion_artifact.test_file_path,
                                        self.data_validation_config.report_file_path
                                         )

            return True

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e


    def get_save_data_drift_report_page(self):
        try:

            last_pushed_raw_path = self.get_last_pushed_raw_path()

            is_last_pushed_exist = len(last_pushed_raw_path) < 1 and \
                                   os.path.exists(last_pushed_raw_path)  

            if not is_last_pushed_exist:
                message = f""" Last pushed train and test is not found. 
                Data drift report only be done on the test and train data
                """
                logging.info(message)

            if is_last_pushed_exist:
                raw_data_file_path = self.data_ingestion_artifact.raw_file_path
                # Compare last pushed raw data with current data
                self.save_data_drift_report_page(raw_data_file_path, 
                                            last_pushed_raw_path,
                                            self.data_validation_config.pushed_report_page_file_name
                                            )
            
            # Compare the train and test data drift
            self.save_data_drift_report_page(self.data_ingestion_artifact.train_file_path,
                                        self.data_ingestion_artifact.test_file_path,
                                        self.data_validation_config.report_page_file_name)
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def is_data_drift_found(self):
        try:
            is_pushed_model_data_drift_found = False
            is_train_test_data_drift_found = False

            report = self.get_and_save_data_drift_reports()
            self.get_save_data_drift_report_page()

            if os.path.exists(self.data_validation_config.pushed_report_file_path):
                pushed_report_json = read_json(self.data_validation_config.pushed_report_file_path)
                is_pushed_model_data_drift_found = bool(pushed_report_json['data_drift']['data']['metrics']['dataset_drift'])
            
            train_test_data_report_json = read_json(self.data_validation_config.report_file_path)
            is_train_test_data_drift_found = bool(train_test_data_report_json['data_drift']['data']['metrics']['dataset_drift'])

            if is_pushed_model_data_drift_found:
                message = f"""
                Data drift found in the last pushed model raw data and the current raw data.
                Report can be found in { self.data_validation_config.pushed_report_file_path } and 
                Report page can be found in {self.data_validation_config.pushed_report_page_file_name}
                """          
                raise Exception(message)
            
            if is_train_test_data_drift_found:
                message = f"""
                Data drift found in the train and test data.
                Report can be found in { self.data_validation_config.report_file_path } and 
                Report page can be found in {self.data_validation_config.report_page_file_name}
                """            
                raise Exception(message)

            logging.info("No drift found in the data")            
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e
            
    def initiate_data_validation(self) -> DataValidationArtifact:

        """
            Description:
            This function used to validate the data

            Args:
               None
            Returns:
                Data validation artifacts
            Raises:
                InsurancePrediction Exception
        """
        try:
            self.is_train_test_file_exist()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path= self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_name,
                pushed_report_file_path=self.data_validation_config.pushed_report_file_path,
                pushed_report_page_file_path=self.data_validation_config.pushed_report_page_file_name,
                is_validated=True,
                message="DataValidation Completed Successfully"
            )

            return data_validation_artifact
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e