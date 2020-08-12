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
tf.disable_v2_behavior()
import time


BATCH_SIZE = 32
IMG_SIZE = 299
TRAIN_SIZE = 1153006
VALID_SIZE = 128161
#
def train_xception(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    xception_model = Xception(
        weights='imagenet',
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
    for layer in model.layers[:-3]:
        layer.trainable = False
    model = tf.keras.models.load_model('./Models/Xception')
    train_df = pd.read_csv('./classes_train.csv')
    validate_df = pd.read_csv('./classes_validate.csv')
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(beta_1=.999),
        metrics=['accuracy']
    )
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file',
        y_col='class_num',
        target_size=(img_size, img_size),
        validate_filenames=False,
        batch_size=batch_size,
        class_mode='categorical'
    )
    validate_generator = train_datagen.flow_from_dataframe(
        dataframe=validate_df,
        x_col='file',
        y_col='class_num',
        validate_filenames=False,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    model.fit(
        train_generator,
        validation_data=validate_generator,
        epochs=10,
        workers=16,
        steps_per_epoch=VALID_SIZE // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('./Models/Xception'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        ]
    )
    # print(model.evaluate(train_generator, steps=40, batch_size=batch_size*4))
    model.save('./Models/Xception')

def main(train=False, batch_size=BATCH_SIZE):
    if train:
        train_xception()
    # noise_in = layers.Input(shape=128)
    # label_in = layers.Input(shape=1000)
    # label_input = layers.Input(shape=1000)
    # noise_input = layers.Input(shape=128)
    # truncation_input = tf.keras.backend.placeholder(shape=())
    # truncation_input = layers.InputLayer(input_tensor=truncation_input)._inbound_nodes[0].output_tensors[0]
    # print(truncation_input)
    # truncation_input = layers.Input(tensor=truncation)
    big_gan_layer = hub.Module('./Models/BigGan_256')
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in big_gan_layer.get_input_info_dict().items()}
    output = big_gan_layer(inputs)

    # xception_layer = hub.KerasLayer('./Models/Xception', input_shape=[], dtype=tf.float32)(big_gan_layer.output)
    # print(big_gan_layer)
    # big_gan_model = tf.keras.Model([label_input, noise_input, truncation_input], big_gan_layer)
    # big_gan_model.summary()
    xception_model = tf.keras.models.load_model('./Models/Xception')
    # model.add)
    # model.add(

    # )
    truncation = .5
    noise = (tf.random.truncated_normal([batch_size, 128]) * truncation)
    label_map = defaultdict(list)
    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)
    print(sess.list_devices())
    noise = noise.eval(session=sess)
    for label in trange(1000):
        with tf.device('/GPU:0'):
            labels = tf.one_hot([label] * batch_size, 1000).eval(session=sess)
            # print(labels.shape)

            samples = sess.run(output, feed_dict={inputs['y']: labels, inputs['z']: noise, inputs['truncation']: truncation})
            # resized_samples = tf.image.resize(samples, (IMG_SIZE, IMG_SIZE))
            # predicted_values = xception_model(resized_samples)
            # predicted_values = tf.reduce_sum(predicted_values, axis=0)
            # # print(predicted_values, label)
            # predicted_values = tf.argmax(predicted_values)
            # label_map[label].append(predicted_values)
    print(label_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn Snake Making Gan labels.')
    parser.add_argument(
        '--train',
        help='Train Xception',
        action='store_true',
        default=False
    )
    # tf.keras.backend.set_epsilon(1e-7)
    args = parser.parse_args()
    main(args.train)