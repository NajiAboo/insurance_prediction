import os
import sys
import shutil
import numpy as np
import pandas as pd

from typing import Tuple
from six.moves import urllib

from insuranceprediction.logger import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from insuranceprediction.entity.artifact_entity import DataIngestionArtifact
from insuranceprediction.entity.config_entity import DataIngestionConfig
from insuranceprediction.exception import InsurancePredictionException


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as ex:
            raise InsurancePredictionException(ex, sys) from ex
        
    
    def download_insurance_premium_data(self) -> str:
        try:
            download_data_url = self.data_ingestion_config.dataset_download_url
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            os.makedirs(raw_data_dir, exist_ok=True)
            insurance_premium_file_name = os.path.basename(download_data_url)

            raw_file_path = os.path.join(raw_data_dir, insurance_premium_file_name)

            logging.info(f"Downloaded the file from [{download_data_url}] into [{raw_file_path}]")
            urllib.request.urlretrieve(download_data_url, raw_file_path)
            logging.info(f"File: [{raw_file_path}] has been downloaded successfully")

            return raw_file_path        

        except Exception as ex:
            raise InsurancePredictionException(ex, sys) from ex

    
    def split_data_train_test(self, raw_data_file_path) -> Tuple:
        try:

            logging.info(f"Reading csv file: [{raw_data_file_path}]")
            df = pd.read_csv(raw_data_file_path)

            file_name = os.path.basename(raw_data_file_path)

            logging.info(f"Creating new cateogry ")
            df["charge_cat"] = pd.cut( df["charges"],
                                bins=[0.0, 10000, 20000, 30000, 40000, 50000, np.inf],
                                labels=[1,2,3,4,5,6]
                            )

            logging.info("Splitting the data into Train and Test Data")
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(df,df["charge_cat"] ):
                strat_train_set = df.loc[train_index].drop(["charge_cat"],axis=1)
                strat_test_set = df.loc[test_index].drop(["charge_cat"],axis=1)


            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=140)

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)
            
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting testing dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)


            return (train_file_path, test_file_path)
        except Exception as ex:
            raise InsurancePredictionException(ex, sys) from ex

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:

            raw_data_file_path = self.download_insurance_premium_data()

            train_file_path, test_file_path = self.split_data_train_test(raw_data_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path= test_file_path,
                raw_file_path=raw_data_file_path,
                is_ingested=True,
                message="Data Ingestion completed successfully")   

            return data_ingestion_artifact

        except Exception as ex:
            raise InsurancePredictionException(ex, sys) from ex