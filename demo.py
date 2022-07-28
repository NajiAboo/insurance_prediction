import sys
import os
from insuranceprediction.config.configuration import Configuration
from insuranceprediction.entity.model_factory import ModelFactory

from insuranceprediction.exception import InsurancePredictionException
from insuranceprediction.logger import logging
from insuranceprediction.pipeline.pipeline import Pipeline


def main():
    try:
       # pipeline = Pipeline()
       # pipeline.run_pipeline()
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        pipeline.start()
       #m_factory =  ModelFactory (f"config/model.yaml")
       
    except Exception as ex:
        #medical_exception = InsurancePredictionException(ex, sys)
        logging.error(f"{ex}")
        print(ex)


if __name__ == "__main__":
    main()