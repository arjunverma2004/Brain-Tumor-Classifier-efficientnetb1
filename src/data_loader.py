import tensorflow as tf
import tensorflow.keras as keras




def data_load(train_path, test_path, IMG_SIZE, BATCH_SIZE):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    num_classes = len(train_ds.class_names)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path, # Corrected from TEST_DIR to test_path
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, test_ds, num_classes
