import os
from pathlib import Path
from typing import List, Optional
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from nephroscan_ai.config.configuration import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: Optional[tf.keras.Model] = None  # pyright: ignore[reportAttributeAccessIssue]
        self.train_generator: Optional[tf.keras.preprocessing.image.DirectoryIterator] = None  # pyright: ignore[reportAttributeAccessIssue]
        self.valid_generator: Optional[tf.keras.preprocessing.image.DirectoryIterator] = None  # pyright: ignore[reportAttributeAccessIssue]
        self.steps_per_epoch: Optional[int] = None
        self.validation_steps: Optional[int] = None
        self.class_weights = None

    def _resolve_training_dir(self, training_dir: Path) -> Path:
        training_dir = Path(training_dir)
        if not training_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {training_dir}")

        subs = [d for d in training_dir.iterdir() if d.is_dir()]
        if len(subs) == 1:
            inner = subs[0]
            inner_subs = [d for d in inner.iterdir() if d.is_dir()]
            if len(inner_subs) > 1:
                print(f"Auto-resolved training_data -> using inner folder: {inner}")
                return inner
        return training_dir

    def get_base_model(self):
        updated = Path(self.config.updated_base_model_path)
        base = updated.parent / "base_model.keras"
        if updated.exists():
            self.model = tf.keras.models.load_model(str(updated))  # pyright: ignore[reportAttributeAccessIssue]
            print(f"✅ Loading updated base model: {updated}")
        elif base.exists():
            self.model = tf.keras.models.load_model(str(base))  # pyright: ignore[reportAttributeAccessIssue]
            print(f"⚠️ Loading base model (not updated): {base}")
        else:
            raise FileNotFoundError(
                f"No model found at {updated} or {base}. Please run Phase 02 (prepare_base_model)."
            )

    def train_valid_generator(self):
        raw_training_dir = Path(self.config.training_data)
        training_dir = self._resolve_training_dir(raw_training_dir)

        # Debug: list class folders
        class_folders: List[str] = sorted([d.name for d in training_dir.iterdir() if d.is_dir()])
        print("Detected class folders:", class_folders)

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=tuple(self.config.params_image_size[:2]),
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(  # pyright: ignore[reportAttributeAccessIssue]
            **datagenerator_kwargs
        )

        # Validation generator
        self.valid_generator = valid_datagenerator.flow_from_directory(  # pyright: ignore[reportAttributeAccessIssue]
            directory=str(training_dir),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator (with augmentation if requested)
        if self.config.params_is_augmented:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(  # pyright: ignore[reportAttributeAccessIssue]
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(  # pyright: ignore[reportAttributeAccessIssue]
            directory=str(training_dir),
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Make static-type-checker + runtime safe before accessing attributes
        assert self.train_generator is not None, "train_generator must be created"
        # Try to get labels robustly (classes preferred, then labels, then infer from filenames)
        labels = getattr(self.train_generator, "classes", None)  # pyright: ignore[reportAttributeAccessIssue]
        if labels is None:
            labels = getattr(self.train_generator, "labels", None)  # pyright: ignore[reportAttributeAccessIssue]

        if labels is None:
            # fallback: infer from filenames + class_indices if available
            class_indices = getattr(self.train_generator, "class_indices", None)  # pyright: ignore[reportAttributeAccessIssue]
            filenames = getattr(self.train_generator, "filenames", None)  # pyright: ignore[reportAttributeAccessIssue]
            if class_indices is not None and filenames is not None:
                labels = np.array([class_indices[f.split(os.sep)[0]] for f in filenames])
            else:
                raise AttributeError(
                    "Unable to determine labels from train_generator. "
                    "Expected 'classes' or 'labels' attribute or inferable 'filenames' + 'class_indices'."
                )

        labels = np.asarray(labels)
        # compute balanced class weights
        classes = np.unique(labels)
        class_weights_array = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        # map to dict expected by Keras
        self.class_weights = {int(cls): float(w) for cls, w in zip(classes, class_weights_array)}
        print("📊 Computed class weights:", self.class_weights)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):  # pyright: ignore[reportAttributeAccessIssue]
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        model.save(str(path))  # pyright: ignore[reportAttributeAccessIssue]
        print(f"✅ Final model saved at {path}")

    def train(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call get_base_model() first.")
        if self.train_generator is None or self.valid_generator is None:
            raise ValueError("Generators not created. Call train_valid_generator() first.")

        self.steps_per_epoch = max(1, self.train_generator.samples // max(1, self.train_generator.batch_size))
        self.validation_steps = max(1, self.valid_generator.samples // max(1, self.valid_generator.batch_size))

        print(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")
        print(f"Train samples: {self.train_generator.samples}, Valid samples: {self.valid_generator.samples}")
        print(f"Model output shape: {self.model.output_shape}")

        # checkpoint filename must end with `.weights.h5`
        checkpoint_path = Path(self.config.root_dir) / "model.weights.h5"
        os.makedirs(checkpoint_path.parent, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(  # pyright: ignore[reportAttributeAccessIssue]
                filepath=str(checkpoint_path),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(  # pyright: ignore[reportAttributeAccessIssue]
                monitor="val_loss",
                factor=0.5,
                patience=2,
                verbose=1,
                min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(  # pyright: ignore[reportAttributeAccessIssue]
                monitor="val_loss",
                patience=5,
                verbose=1,
                restore_best_weights=True
            )
        ]

        history = self.model.fit(  # pyright: ignore[reportAttributeAccessIssue]
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.save_model(path=Path(self.config.trained_model_path), model=self.model)
        return history
