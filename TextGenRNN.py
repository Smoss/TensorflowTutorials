import tensorflow as tf

import numpy as np
import os
import time
import tqdm
import tensorflow_datasets as tfds
import functools
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])
# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
# The maximum length sentence we want for a single input in characters
seq_length = 500
examples_per_epoch = len(text)//(seq_length+1)

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
print('Mask is ', encoder.decode([0]))
print(train_data)
# train_data, _ = train_data
# test_data, _ = test_data

# Create training examples / targets
char_dataset = train_data# tf.data.Dataset.from_tensor_slices(train_data)

# for i in char_dataset.take(5):
#   print(idx2char[i.numpy()])
# sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# for item in char_dataset.take(5):
#   print(repr(''.join(idx2char[item.numpy()])))
max_length = 0
dataset = char_dataset
for input, target in dataset:
    # for input in input_batch:
    max_length = max(len(input), max_length)
    # print(input)
    # temp_inputs.append(input)
max_length = 512
def split_input_target(chunk, target):
    mask_size = max_length - len(chunk)
    if mask_size >= 0:
        mask = tf.zeros((mask_size,), dtype=tf.int64)
        input_text = tf.concat([chunk[:-1], mask], axis=0)
        target_text = tf.concat([chunk[1:], mask], axis=0)
    else:
        input_text = chunk[:max_length]
        target_text = chunk[1:max_length + 1]
    # assert len(input_text) == len(target_text)
    return input_text, target_text

dataset = char_dataset
# dataset = tf.keras.preprocessing.sequence.pad_sequences(dataset)
# sequences = dataset# char_dataset.batch(seq_length+1, drop_remainder=True)
# temp_inputs = []
# max_length = 0
# for input, target in dataset:
#     # for input in input_batch:
#     max_length = max(len(input), max_length)
#     # print(input)
#     temp_inputs.append(input)
# temp_inputs = tf.keras.preprocessing.sequence.pad_sequences(temp_inputs, maxlen=500)
# print(temp_inputs)
# dataset = tf.data.Dataset.from_tensor_slices(temp_inputs).map(split_input_target)
#
# for item in sequences.take(5):
#   print(repr(''.join(idx2char[item.numpy()])))

dataset = dataset.map(split_input_target, num_parallel_calls=16)
for input_example, target_example in dataset.take(1):
    # print(input_example)
    # print(target_example)
    print('Input data: ', repr(''.join(encoder.decode(input_example[input_example > 0]))))
    print('Target data:', repr(''.join(encoder.decode(target_example[target_example > 0]))))
# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print("Step {:4d}".format(i))
#     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
#     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch size
BATCH_SIZE = 16

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 25000

dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, drop_remainder=True)

# print('And the max length is ', max_length)

print(dataset)

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None],
                              mask_zero=True),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(.001)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
  vocab_size = encoder.vocab_size,#len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './Models/GruGenerator'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)

EPOCHS=10
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
# model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
#
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#
# model.build(tf.TensorShape([1, None]))

# print(encoder.subwords[0])
def generate_text(model, start_string):
    num_generate = 1000
    
    # input_eval = [encoder.encode(s) for s in start_string]
    input_eval = encoder.encode(start_string)
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    
    temperature = 1.0
    
    model.reset_states()
    print(input_eval)
    
    for i in range(num_generate):
        # print(i)
        # print(encoder.decode(input_eval.numpy()[0]))
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        predictions = predictions / temperature
        # print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        if predicted_id == 0:
            break
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(encoder.decode([predicted_id]))

    return (start_string + ''.join(text_generated))

# print(generate_text(model, start_string=u"ROMEO: "))

vocab_size = encoder.vocab_size#len(vocab),
model = build_model(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target,
                predictions,
                from_logits=True
            )
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, predictions

# Training step
# EPOCHS = 50
# Directory where the checkpoints will be saved
checkpoint_dir = './Models/GruGeneratorMovie'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True)
# print(vocab_size)
# print(type(vocab_size))
# model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# print(generate_text(model, start_string=u'''Shrek is the voice of a generation'''))
for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for (batch_n, (inp, target)) in tqdm.tqdm(enumerate(dataset), total=(BUFFER_SIZE//BATCH_SIZE)):
        # padded_inp = tf.keras.preprocessing.sequence.pad_sequences(inp, maxlen=max_length)
        # padded_target = tf.keras.preprocessing.sequence.pad_sequences(target, maxlen=max_length)
        loss, predictions = train_step(inp, target)
        epoch_loss_avg.update_state(loss)
        epoch_accuracy.update_state(target, predictions)
        # if batch_n % 1000 == 0:
        #     print(loss)

    template = 'Epoch {} Loss {} Accuracy {}'
    print(template.format(epoch+1, epoch_loss_avg.result(), epoch_accuracy.result()))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
print(generate_text(model, start_string=u'''Shrek is the voice of a generation'''))