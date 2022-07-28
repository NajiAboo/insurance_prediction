import json
from typing import List
import yaml
import os, sys
import csv
import pandas as pd
import numpy as np
import dill

from insuranceprediction.constants import *
from insuranceprediction.exception import InsurancePredictionException

def read_yaml(file_path:str) -> dict:
    try:

        with open(file_path, "rb") as yaml_file:
            config_info = yaml.safe_load(yaml_file)
        return config_info
    except Exception as ex:
        raise InsurancePredictionException(ex,sys) from ex

def read_json(file_path:str) -> dict:
    try:
        with open(file_path) as json_file:
            data = json.load(json_file)
        return data
    except Exception(e) as e:
        raise InsurancePredictionException(e) from e

def load_data_path(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        datatset_schema = read_yaml(schema_file_path)

        schema = datatset_schema[DS_COLUMNS_INFO]

        dataframe = pd.read_csv(file_path)

        error_messgae = ""


        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise InsurancePredictionException(e,sys) from e

def load_data(file_path:str, dataset_schema: dict)-> pd.DataFrame:
        try:
            schema = dataset_schema[DS_COLUMNS_INFO]
            dataframe = pd.read_csv(file_path)

            error_message = ""

            for column in dataframe.columns:
                if column in list(schema.keys()):
                    dataframe[column].astype(schema[column])
                else:
                    error_message = f"{error_message}\n column: [{column}] is not in the schema"
            
            if len(error_message) > 0:
                raise Exception(error_message)
            
            return dataframe

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e



def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise InsurancePredictionException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise InsurancePredictionException(e, sys) from e


def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise InsurancePredictionException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise InsurancePredictionException(e,sys) from e


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise InsurancePredictionException(e,sys)


def write_csv(file_path:str, data: List):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)

    except Exception as e:
        raise InsurancePredictionException(e,sys) from e
