from collections import namedtuple
from threading import Thread
import uuid

from datetime import datetime

import pandas as pd
from insuranceprediction.components.data_transformation import DataTransformation
from insuranceprediction.components.data_validation import DataValidation
from insuranceprediction.components.model_pusher import ModelPusher
from insuranceprediction.components.model_trainer import ModelTrainer
from insuranceprediction.components.data_evaluation import ModelEvaluation
from insuranceprediction.entity.config_entity import DataTransformationConfig
from insuranceprediction.logger import logging
import os
import sys
from insuranceprediction.components.data_ingestion import DataIngestion

from insuranceprediction.config.configuration import Configuration
from insuranceprediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact
from insuranceprediction.exception import InsurancePredictionException
from insuranceprediction.constants import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME


Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])

class Pipeline(Thread):
    experiment = Experiment(*([None] * 11))
    experiment_file_path = None
    

    def __init__(self, config:Configuration = Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            self.config = config
            super().__init__(daemon=False, name="pipeline")
        except Exception as e:
            raise InsurancePredictionException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact
        except Exception as ex:
            raise InsurancePredictionException(ex, sys) from ex

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
           data_validation = DataValidation(data_validation_config=self.config.data_validation_config(), 
                                            data_ingestion_artifact=data_ingestion_artifact)
           data_validation_artifact = data_validation.initiate_data_validation()

           return data_validation_artifact

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                  data_validation_artifiact: DataValidationArtifact,
                                 
                                ) -> DataTransformationArtifact:
        try:
            data_transformation_config: DataTransformationConfig = self.config.get_data_transformation_config()
            
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                              data_validation_artifact=data_validation_artifiact,
                                                              data_transformation_config=data_transformation_config                                                              
                                                             )
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise InsurancePredictionException(e,sys)

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                        data_transformation_artifact=data_transformation_artifact
                                        )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e

    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, 
                            model_transformation_artifact: DataTransformationArtifact, 
                            data_ingestion_artifact: DataIngestionArtifact,
                            data_validation_artifact: DataValidationArtifact
                            ) -> ModelEvaluationArtifact:
        try:
            data_evaluation_config = self.config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(model_evauation_config=data_evaluation_config,
                            data_transformation_artifact=model_transformation_artifact,
                            model_trainer_artifact=model_trainer_artifact,
                            data_ingestion_artifact = data_ingestion_artifact,
                            data_validation_artifact=data_validation_artifact                            
                            )
            model_eval_artifact = model_evaluation.initiate_model_evaluation()

            return model_eval_artifact

        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def start_model_pusher(self,model_eva_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher_config = self.config.get_model_pusher_config()
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config, model_eval_artifact=model_eva_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise InsurancePredictionException(e,sys) from e

    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]})

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("First start experiment")
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e

    
    def get_experiments_status(self, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e             

    def run(self):
            try:
                self.run_pipeline()
            except Exception as e:
                raise e

    def run_pipeline(self):
        try:

            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # data ingestion
            logging.info("Pipeline starting.")

            experiment_id = str(uuid.uuid4())

            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                                initialization_timestamp=self.config.time_stamp,
                                                artifact_time_stamp=self.config.time_stamp,
                                                running_status=True,
                                                start_time=datetime.now(),
                                                stop_time=None,
                                                execution_time=None,
                                                experiment_file_path=Pipeline.experiment_file_path,
                                                is_model_accepted=None,
                                                message="Pipeline has been started.",
                                                accuracy=None,
                                                )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()

            logging.info("Started the pipeline")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Dataingestion artifact = {data_ingestion_artifact}")
            logging.info(f"\nCompleted  data ingestion")

            logging.info("Started the data validation ")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info(f"Data validation artifact = {data_validation_artifact}")
            logging.info(f"\n Completed data validation")   

            logging.info(f"Started data transformation")
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifiact=data_validation_artifact
                                                                        )
            logging.info(f"Data Transformation artifact = {data_transformation_artifact}")
            logging.info(f"\n Completed data transformation")   

            logging.info(f"Started the training")
            model_training_artifact = self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Model training artificact : {model_training_artifact}")  

            logging.info(f'Stated model evaluation')
            model_evaluation_artifact: ModelEvaluationArtifact = self.start_model_evaluation(model_trainer_artifact=model_training_artifact,
                                                                    model_transformation_artifact=data_transformation_artifact,
                                                                    data_ingestion_artifact = data_ingestion_artifact,
                                                                    data_validation_artifact = data_validation_artifact
                                                                    )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")

            if model_evaluation_artifact.is_model_accepted:

                logging.info(f"Started model pusher")
                model_pusher_artifact = self.start_model_pusher(model_eva_artifact=model_evaluation_artifact)
                logging.info(f"Model pusher artifact : {model_pusher_artifact}")
            else:
                logging.info(f"Trained model rejected")
            logging.info(f"Pipleline completed")

            stop_time = datetime.now()
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=False,
                                             start_time=Pipeline.experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - Pipeline.experiment.start_time,
                                             message="Pipeline has been completed.",
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                             accuracy=model_training_artifact.model_accuracy
                                             )

            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
        except Exception as e:
            raise InsurancePredictionException(e, sys) from e