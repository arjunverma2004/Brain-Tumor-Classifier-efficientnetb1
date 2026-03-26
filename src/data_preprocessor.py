import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import efficientnet, EfficientNetB1




augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(15.0/360.0),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1, 0.1),
])

preprocess_input = efficientnet.preprocess_input

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = augmentation(images, training=True)
    images = efficientnet.preprocess_input(images)
    return images, labels

def preprocess_eval(images, labels): # Defined preprocess_eval for test data without augmentation
    images = tf.cast(images, tf.float32)
    images = efficientnet.preprocess_input(images)
    return images, labels