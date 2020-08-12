import argparse

import numpy as py
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tqdm import trange
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import ImageNetLabels
from collections import defaultdict
from PIL import Image
tf.disable_v2_behavior()
import time


BATCH_SIZE = 32
IMG_SIZE = 128
TRAIN_SIZE = 1153006
VALID_SIZE = 128161
#
# def train_xception(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
#     xception_model = Xception(
#         weights='imagenet',
#         input_shape=(img_size, img_size, 3),
#         include_top=False
#     )
#     xception_model.summary()
#     x = layers.GlobalAveragePooling2D()(xception_model.output)
#     x = layers.Flatten()(x)
#     x = layers.Dense(512)(x)
#     x = layers.Dense(512)(x)
#     x = layers.Dense(1000)(x)
#     model = tf.keras.Model(xception_model.input, x)
#     model.summary()
#     # model = tf.keras.models.load_model('./Models/Xception')
#     train_df = pd.read_csv('./classes_train.csv')
#     validate_df = pd.read_csv('./classes_validate.csv')
#     model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
#     train_datagen = ImageDataGenerator(
#         rescale=1. / 255
#     )
#     train_generator = train_datagen.flow_from_dataframe(
#         dataframe=train_df,
#         x_col='file',
#         y_col='class_num',
#         target_size=(img_size, img_size),
#         validate_filenames=False,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )
#     validate_generator = train_datagen.flow_from_dataframe(
#         dataframe=validate_df,
#         x_col='file',
#         y_col='class_num',
#         validate_filenames=False,
#         target_size=(img_size,img_size),
#         batch_size=batch_size,
#         class_mode='categorical'
#     )
#     model.fit(
#         train_generator,
#         epochs=20,
#         workers=16,
#         steps_per_epoch=TRAIN_SIZE // batch_size,
#         callbacks=[tf.keras.callbacks.ModelCheckpoint('./Models/Xception')]
#     )
#     model.evaluate(validate_generator)
#     model.save('./Models/Xception')

def main(batch_size=BATCH_SIZE):
    truncation = tf.constant(.5)
    big_gan_layer = hub.Module('./Models/BigGan_512')

    noise = tf.random.truncated_normal([batch_size, 128]) * truncation
    label_map = defaultdict(list)

    labels = tf.one_hot([487] * batch_size, 1000)

    start = time.time()
    samples = big_gan_layer(dict(y=labels, z=noise, truncation=truncation))
    print(time.time() - start)
    print(samples.shape)
    for idx in trange(samples.shape[0]):
        print(idx)
        img = samples[idx]
        print(img.shape)
        array_to_img(samples[idx]).save('./GenedImages/{}_{}.png'.format(487, idx))
    # print(label_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn Snake Making Gan labels.')
    args = parser.parse_args()
    main()