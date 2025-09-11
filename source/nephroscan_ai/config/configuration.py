from nephroscan_ai.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from nephroscan_ai.utils.common import read_yaml, create_directories
from nephroscan_ai.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig
)
import os
from pathlib import Path


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        # ✅ Use YAML paths directly
        create_directories([config.root_dir])

        prepare_base_model_cfg = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_imagesize=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_classes=params.CLASSES,
            params_weights=params.WEIGHTS
        )

        return prepare_base_model_cfg

    def get_training_config(self) -> TrainingConfig:
        config = self.config.model_training
        params = self.params

        # ✅ Always point to the dataset root, not just unzip dir
        training_data_path = Path(config.training_data)

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            training_data=training_data_path,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmented=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )
        return training_config
