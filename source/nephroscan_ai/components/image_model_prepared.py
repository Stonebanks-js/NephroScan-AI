import os 
import tensorflow as tf
from pathlib import Path
from keras.applications import VGG16
from nephroscan_ai.config.configuration import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        """
        Load the base VGG16 model with given parameters and save it.
        """
        self.model = VGG16(
            input_shape=tuple(self.config.params_imagesize),
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            classes=self.config.params_classes,
        )

        # ensure directory exists
        os.makedirs(os.path.dirname(self.config.base_model_path), exist_ok=True)
        self.model.save(self.config.base_model_path)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare a custom model on top of the base VGG16.
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output) # pyright: ignore[reportAttributeAccessIssue]
        prediction = tf.keras.layers.Dense( # pyright: ignore[reportAttributeAccessIssue]
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model( # pyright: ignore[reportAttributeAccessIssue]
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), # pyright: ignore[reportAttributeAccessIssue]
            loss=tf.keras.losses.CategoricalCrossentropy(), # pyright: ignore[reportAttributeAccessIssue]
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Freeze layers and update the base model with classification head.
        """
        if self.model is None:
            raise ValueError("Base model not loaded. Run get_base_model() first.")

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        # ensure directory exists
        os.makedirs(os.path.dirname(self.config.updated_base_model_path), exist_ok=True)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    def save_model(self, path: Path, model: tf.keras.Model): # pyright: ignore[reportAttributeAccessIssue]
        """
        Save the given model to the provided path.
        """
        model.save(path)
