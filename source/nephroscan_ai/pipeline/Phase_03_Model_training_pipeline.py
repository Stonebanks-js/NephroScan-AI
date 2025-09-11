# source/nephroscan_ai/pipeline/Phase_03_Model_training_pipeline.py
from pathlib import Path
import importlib
import logging

import tensorflow as tf  # pyright: ignore[reportUndefinedVariable]
from nephroscan_ai.config.configuration import ConfigurationManager
from nephroscan_ai.components.model_training import Training
from nephroscan_ai import logger

STAGE_NAME = "Model Training Stage"


def _load_model_into_training(training_obj, updated_path: Path, base_path: Path):
    """
    Load model into training_obj.model with safety and logging.
    """
    if updated_path.exists():
        logger.info(f"Loading updated model from {updated_path}")
        training_obj.model = tf.keras.models.load_model(str(updated_path))  # pyright: ignore[reportAttributeAccessIssue]
    elif base_path.exists():
        logger.info(f"Loading base model from {base_path}")
        training_obj.model = tf.keras.models.load_model(str(base_path))  # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise FileNotFoundError(
            f"No model found at {updated_path} or {base_path}. Please re-run Phase 02 (prepare_base_model)."
        )


def _try_fast_training(training_cfg):
    """
    Attempt to import fast_training.run_fast_training and execute it.
    The fast_training.py should expose a function with signature:
        def run_fast_training(training_cfg: TrainingConfig) -> None
    If found and runs, return True. If missing or error, log and return False.
    """
    try:
        fast_mod = importlib.import_module("fast_training")
    except Exception as e:
        logger.info("fast_training module not found; will use modular Training component. "
                    f"(import error: {e})")
        return False

    run_fn = getattr(fast_mod, "run_fast_training", None)
    if run_fn is None:
        # Maybe the fast_training script exposes a top-level run() or main(); try common names
        run_fn = getattr(fast_mod, "main", None) or getattr(fast_mod, "run", None)

    if run_fn is None:
        logger.info("fast_training module found but no run_fast_training/main/run function; falling back.")
        return False

    # Call the fast training function (it should accept training_cfg or read config internally).
    try:
        logger.info("Running fast_training.run_fast_training(...)")
        # If the function expects a TrainingConfig, pass it; otherwise call without args.
        try:
            run_fn(training_cfg)
        except TypeError:
            run_fn()
        logger.info("fast_training completed successfully.")
        return True
    except Exception as e:
        logger.exception(f"fast_training failed: {e}")
        return False


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Prefer fast_training flow if available
        fast_success = _try_fast_training(training_config)
        if fast_success:
            logger.info("Phase 03 completed using fast_training.")
            return

        # otherwise use modular Training component (head -> fine-tune as implemented)
        training = Training(config=training_config)

        # Resolve model paths (string paths just in case)
        updated_model_path = Path(training_config.updated_base_model_path)
        base_model_path = Path(updated_model_path.parent) / "base_model.keras"

        # Load model into training object (this avoids re-running PrepareBaseModel)
        _load_model_into_training(training, updated_model_path, base_model_path)

        # create train/valid generators
        training.train_valid_generator()

        # train (this will save final model into training_config.trained_model_path)
        training.train()


if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
