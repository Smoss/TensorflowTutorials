import numpy as py

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
embedding = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'
# train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(
    optimizer='nadam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
BATCH_SIZE = 10000
history = model.fit(
    train_data.shuffle(10000).batch(BATCH_SIZE),
    epochs=40,
    validation_data=validation_data.batch(BATCH_SIZE),
    verbose=1
)

results = model.evaluate(test_data.batch(BATCH_SIZE), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
