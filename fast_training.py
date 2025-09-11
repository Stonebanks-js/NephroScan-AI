# fast_training.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf  # pyright: ignore
from tensorflow.keras import layers, models, optimizers  # pyright: ignore
from sklearn.utils.class_weight import compute_class_weight  # pyright: ignore

# local imports will be done by caller, or we accept passed configs
def run_fast_training(cfg=None,
                      training_cfg=None,
                      head_epochs: int = 5,
                      finetune_epochs: int = 5,
                      batch_size_override: int | None = None,
                      quick_steps: int | None = None,
                      enable_mixed_precision: bool = True):
    """
    Run a fast two-stage training:
      - Stage A: train head only (backbone frozen)
      - Stage B: optional quick finetune (unfreeze some layers)
    Parameters:
      - cfg: optional ConfigurationManager instance (not required if training_cfg provided)
      - training_cfg: TrainingConfig instance (required)
      - head_epochs, finetune_epochs: ints
      - batch_size_override: if provided use this batch size instead of training_cfg.params_batch_size
      - quick_steps: if provided, cap steps_per_epoch to this number (useful to speed runs)
    Returns:
      history_head, history_finetune (finetune may be None if skipped)
    """
    if training_cfg is None:
        if cfg is None:
            raise ValueError("Provide either cfg or training_cfg")
        training_cfg = cfg.get_training_config()

    IMG_SIZE = tuple(training_cfg.params_image_size[:2])
    BATCH_SIZE = int(batch_size_override or training_cfg.params_batch_size)
    EPOCHS_HEAD = max(1, int(head_epochs))
    EPOCHS_FINETUNE = max(0, int(finetune_epochs))

    # ----- data generators -----
    datagen_args = dict(rescale=1.0 / 255.0, validation_split=0.20)
    train_gen_obj = tf.keras.preprocessing.image.ImageDataGenerator(   # type: ignore
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        **datagen_args
    )
    val_gen_obj = tf.keras.preprocessing.image.ImageDataGenerator( # type: ignore
        rescale=1.0 / 255.0, validation_split=0.20
    )

    train_gen = train_gen_obj.flow_from_directory(
        directory=str(training_cfg.training_data),
        subset="training",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        interpolation="bilinear",
        shuffle=True,
        class_mode="categorical"
    )

    val_gen = val_gen_obj.flow_from_directory(
        directory=str(training_cfg.training_data),
        subset="validation",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode="categorical"
    )

    num_classes = getattr(train_gen, "num_classes", None) or train_gen.class_indices and len(train_gen.class_indices) or None
    if num_classes is None:
        raise RuntimeError("Could not determine number of classes from training generator")
    print("num_classes:", num_classes)
    train_samples = train_gen.samples
    val_samples = val_gen.samples
    print(f"Train samples: {train_samples} Val samples: {val_samples}")

    # calculate class weights (helps imbalance)
    labels = train_gen.classes  # numpy array of labels
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=labels)
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("class_counts:", list(class_counts))
    print("class_weights:", class_weights_dict)

    # ----- model (MobileNetV2 backbone + small head) -----
    base = tf.keras.applications.MobileNetV2(  # type: ignore
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # freeze backbone for Stage A

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ----- callbacks & checkpoint filenames -----
    ckpt_dir = Path(training_cfg.root_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_path_head = ckpt_dir / "fast_model.weights.h5"
    weights_path_finetune = ckpt_dir / "fast_model_finetuned.weights.h5"
    final_model_path = Path(training_cfg.trained_model_path)

    callbacks_head = [
        tf.keras.callbacks.ModelCheckpoint(str(weights_path_head), monitor='val_loss',  # type: ignore
                                           save_best_only=True, save_weights_only=True, verbose=1),  # type: ignore
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),  # type: ignore
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)  # type: ignore
    ]

    # optional mixed precision
    if enable_mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16') # type: ignore
            tf.keras.mixed_precision.set_global_policy(policy) # type: ignore
            print("Mixed precision enabled:", policy)
        except Exception:
            pass

    # steps tuning (quick debug vs full)
    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)
    if quick_steps:
        steps_per_epoch = min(quick_steps, steps_per_epoch)
        validation_steps = min(max(1, quick_steps // 4), validation_steps)

    print("Using steps_per_epoch=", steps_per_epoch, "validation_steps=", validation_steps)

    # ----- Stage A: train head -----
    print("Training head (backbone frozen)...")
    history_head = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_head,
        class_weight=class_weights_dict,
        verbose=1
    )

    # save head model
    out_head = ckpt_dir / "fast_model.keras"
    model.save(str(out_head))
    print("Saved fast model (head):", out_head)

    # ----- Stage B: optional finetune -----
    history_finetune = None
    if EPOCHS_FINETUNE > 0:
        print("Starting fine-tuning stage...")
        base.trainable = True
        # freeze most layers except last N (here keep last 30 trainable)
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks_ft = [
            tf.keras.callbacks.ModelCheckpoint(str(weights_path_finetune), monitor='val_loss', # type: ignore
                                               save_best_only=True, save_weights_only=True, verbose=1), # type: ignore
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), # type: ignore
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2) # type: ignore
        ]

        history_finetune = model.fit(
            train_gen,
            epochs=EPOCHS_FINETUNE,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_ft,
            class_weight=class_weights_dict,
            verbose=1
        )

        # save finetuned model
        out_ft = ckpt_dir / "fast_model_finetuned.keras"
        model.save(str(out_ft))
        print("Saved finetuned model:", out_ft)

    # final save
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print("Saved final fast model:", final_model_path)

    return history_head, history_finetune
