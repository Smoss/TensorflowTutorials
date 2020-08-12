import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure.
    with_info=True
)
encoder = info.features['text'].encoder
print('Vocab size {}'.format(encoder.vocab_size))
sample_string = 'Hello TensorFlow'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in encoded_string:
    print('{} => {}'.format(ts, encoder.decode([ts])))

for train_example, train_label in train_data.take(1):
    print('Encoded text:', train_example[:10].numpy())
    print('Label:', train_label.numpy())

    encoder.decode(train_example)

BUFFER_SIZE = 25000
BATCH_SIZE = 128

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE))

test_batches = (
    test_data
    .padded_batch(BATCH_SIZE))

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    # keras.layers.Conv1D(16, 3, padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dropout(.5),
    # keras.layers.Conv1D(32, 3, padding='same'),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dropout(.5),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32),
    keras.layers.Dense(1)#, kernel_regularizer=keras.regularizers.l2())
])
model.compile(
    optimizer=tf.optimizers.Adam(.001),
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.summary()
history = model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=30)

loss, accuracy = model.evaluate(test_batches)
print("Loss: ", loss)
print("Accuracy:", accuracy)

history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
