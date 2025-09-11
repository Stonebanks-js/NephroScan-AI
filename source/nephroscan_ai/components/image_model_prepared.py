# source/nephroscan_ai/components/image_model_prepared.py
import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import mobilenet_v2  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
from nephroscan_ai.config.configuration import PrepareBaseModelConfig

# pyright: ignore[reportMissingTypeStubs]
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        """
        Create MobileNetV2 base (include_top=False) and save base model file.
        """
        self.model = mobilenet_v2.MobileNetV2(
            input_shape=tuple(self.config.params_imagesize),
            include_top=False,
            weights=self.config.params_weights
        )  # pyright: ignore

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(self.config.base_model_path), exist_ok=True)
        self.model.save(str(self.config.base_model_path))  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model,  # type: ignore
                            classes: int,
                            freeze_all: bool,
                            freeze_till: int | None,
                            learning_rate: float,
                            dense_units: int = 512,
                            dropout_rate: float = 0.5) -> tf.keras.Model:  # type: ignore
        """
        Compose classification head on top of base MobileNetV2.
        Head: GAP -> Dense(dense_units, relu) -> Dropout -> Dense(classes, softmax)
        """
        # freeze layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)  # pyright: ignore
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)  # pyright: ignore
        x = tf.keras.layers.Dropout(dropout_rate)(x)  # pyright: ignore
        out = tf.keras.layers.Dense(units=classes, activation="softmax")(x)  # pyright: ignore

        full_model = tf.keras.models.Model(inputs=model.input, outputs=out)  # pyright: ignore
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),  # type: ignore
            loss=tf.keras.losses.CategoricalCrossentropy(),  # type: ignore
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self,
                          freeze_all: bool = True,
                          freeze_till: int | None = None,
                          learning_rate: float | None = None,
                          dense_units: int = 512,
                          dropout_rate: float = 0.5):
        """
        Build and save the updated base model (with head).
        """
        if self.model is None:
            raise ValueError("Base model not loaded. Run get_base_model() first.")

        lr = learning_rate if learning_rate is not None else self.config.params_learning_rate

        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=freeze_all,
            freeze_till=freeze_till,
            learning_rate=lr,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(self.config.updated_base_model_path), exist_ok=True)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    def save_model(self, path: Path, model: tf.keras.Model):  # pyright: ignore[reportAttributeAccessIssue]
        model.save(str(path))  # pyright: ignore[reportAttributeAccessIssue]
