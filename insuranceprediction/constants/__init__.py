import os
from datetime import datetime
from this import d

from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
ROOT_DIR = os.getcwd()

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"

CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)
LAST_PUSHED_MODEL_CONFIG_INFO ="pushed_artifact_info.yaml"


CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

#Training pipeline 
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

#data ingestion related varaible
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_ROW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGECTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

DATA_VALIDATION_CONFIG_KEY = "data_valiadtion_config"
DATA_VALIDATION_ARTIFACT_DIR = 'data_validation'
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_PUSHED_FILE_NAME_KEY = "pushed_report_file_name"
DATA_VALIDATION_PUSHED_REPORT_PAGE_FILE_NAME_KEY = "pushed_report_page_file_name"


DATA_TRAINSFORMED_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_CONFIG_KEY ="data_transformation_config"
DATA_TRANSFORMED_DIR_KEY = "transformed_dir"
DATA_TRANSFORMED_TRAIN_DIR_KEY ="transformed_train_dir"
DATA_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMED_PREPROCESSING_DIR_KEY ="preprocessing_dir"
DATA_TRANSFORMED_PREPROCESSED_OBJECT_FILE_NAME = "preprocessed_object_file_name"
 

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
#MODEL_TRAINER_ARTIFACT= "trained_model_dir"
MODEL_TRAINER_CONFIG_KEY= "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model"
MODEL_TRAINER_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "best_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY= "model_config_file_name"

MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"
MODEL_EVALUATION_HISTORY_FILE_PATH ="model_evaluation_history_file_path"

MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

NUMERICAL_COLUMNS = "numerical_columns"
CATEGORICAL_COLUMNS = "categorical_columns"

#Schema.yaml constants
DS_COLUMNS_INFO = "columns"
DS_CATEGORICAL_COLUMNS = "categorical_columns"
DS_DOMAIN_VALUES ="domain_values"
DS_TARGET_COLUMN_NAME = "target_column"

BEST_MODEL_KEY = "best_model"
MODEL_PATH_KEY = "model_path"
HISTORY_HEADER = ["datetime","model_path"]

PUSHED_MODEL_INGESTION_TRAIN_PATH = "pushed_model_ingestion_train_path"
PUSHED_MODEL_INGESTION_TEST_PATH = "pushed_model_ingestion_test_path"
PUSHED_MODEL_INGESTION_RAW_PATH = "pushed_model_ingestion_raw_path"


EXPERIMENT_DIR_NAME="experiment"
EXPERIMENT_FILE_NAME="experiment.csv"

