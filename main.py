from nephroscan_ai import logger
from nephroscan_ai.pipeline.Phase_01_Data_ingestion_pipeline import data_ingestion_pipeline
from nephroscan_ai.pipeline.Phase_02_Image_model_training_pipeline import PrepareBaseModelTrainingPipeline
from nephroscan_ai.pipeline.Phase_03_Model_training_pipeline import ModelTrainingPipeline

# ==============================
# STAGE 01: Data Ingestion
# ==============================
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = data_ingestion_pipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


# ==============================
# STAGE 02: Prepare Base Model
# ==============================
STAGE_NAME = "Prepare base model"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# ==============================
# STAGE 03: Model Training
# ==============================
STAGE_NAME = "Model Training"

try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
