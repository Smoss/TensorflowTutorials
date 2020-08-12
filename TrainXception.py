import numpy as py

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tqdm import trange
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import pandas as pd

BATCH_SIZE = 32
IMG_SIZE = 128

def main():

    xception_model = Xception(
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False
    )
    xception_model.summary()
    x = layers.GlobalAveragePooling2D()(xception_model.output)
    x = layers.Flatten()(x)
    x = layers.Dense(512)(x)
    x = layers.Dense(512)(x)
    x = layers.Dense(1000)(x)
    model = tf.keras.Model(xception_model.input, x)
    model.summary()
    train_df = pd.read_csv('./classes_train.csv')
    validate_df = pd.read_csv('./classes_validate.csv')
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file',
        y_col='class_num',
        target_size=(IMG_SIZE, IMG_SIZE),
        validate_filenames=False,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    validate_generator = train_datagen.flow_from_dataframe(
        dataframe=validate_df,
        x_col='file',
        y_col='class_num',
        validate_filenames=False,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    model.fit(train_generator, validation_data=validate_generator, epochs=10, workers=16)
    model.save('./Models/Xception')

if __name__ == "__main__":
    main()