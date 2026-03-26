import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.data import AUTOTUNE
from tensorflow.keras.applications import efficientnet, EfficientNetB1
from src import data_ingestion, data_loader, utils


def train_model(train_ds, test_ds, IMG_SIZE=(224,224), num_classes=4):

    base_model = EfficientNetB1(include_top=False, input_shape=(*IMG_SIZE, 3), weights="imagenet")
    base_model.trainable = True


    print(f"Total layers in base model: {len(base_model.layers)}")

    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x =(base_model(inputs, training=True))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    # final Dense with softmax for categorical labels
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    #Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_auc", mode="max")
    reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    earlystop_cb = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=[checkpoint_cb, reduce_lr_cb, earlystop_cb]
    )

    test_loss, test_acc, test_auc = model.evaluate(test_ds)

    print(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}, test auc: {test_auc:.4f}")
    

    return model


if __name__=="__main__":


    IMG_SIZE=(224,224)
    BATCH_SIZE=64
    train_path="/kaggle/input/brain-tumor-mri-dataset/Training"
    test_path="/kaggle/input/brain-tumor-mri-dataset/Testing"

    train_ds, test_ds, num_classes = data_loader.data_load(train_path, test_path, IMG_SIZE, BATCH_SIZE)
    train_ds, test_ds = data_ingestion.get_data(train_ds, test_ds)

    model = train_model(train_ds, test_ds, IMG_SIZE, num_classes)
    utils.save_model(model)