from collections import namedtuple
from this import d


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataIngestionConfig = namedtuple("DataIngestionConfig",
                    ["dataset_download_url", "raw_data_dir","ingested_train_dir","ingested_test_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                    ["schema_file_path",
                    "report_file_path",
                    "report_page_file_name",
                    "pushed_report_file_path",
                    "pushed_report_page_file_name"
                    ]
                    )

DataTransformationConfig =namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path"])

DataModelTrainerConfig = namedtuple("DataModelTrainerConfig", 
                                   ["trained_model_file_path","base_accuracy","model_config_file_path"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path","time_stamp","model_evaluation_history_file_path"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])
