import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
img_size = 299
out_size = 149
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

flip = tf.constant(0.5)

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (img_size, img_size))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (out_size, out_size))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (img_size, img_size))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (out_size, out_size))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
# display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

# base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
#
# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',   # 64x64
#     'block_3_expand_relu',   # 32x32
#     'block_6_expand_relu',   # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',      # 4x4
# ]

class ConvBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            channels,
            use_bias=True,
            kernel=3,
            stride=2,
            padding='same',
            momentum=.9,
            name=None,
            upscale=False,
            transpose=False,
            **kwargs
    ):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.batchNorm = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.ReLU = tf.keras.layers.ReLU()
        if transpose:
            self.conv = tf.keras.layers.Conv2DTranspose(
                channels,
                kernel,
                stride,
                padding=padding,
                use_bias=use_bias,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4)
            )
        else:
            self.conv = tf.keras.layers.SeparableConv2D(
                channels,
                kernel,
                stride,
                padding=padding,
                use_bias=use_bias,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                bias_regularizer=tf.keras.regularizers.l2(1e-4)
            )
        self.upscale = upscale
        if upscale:
            self.upsample = tf.keras.layers.UpSampling2D()

    def call(self, inputs):
        output_t = self.batchNorm(inputs)
        output_t = self.ReLU(output_t)
        if self.upscale:
            output_t = self.upsample(output_t)
        output_t = self.conv(output_t)

        return output_t

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update()
        return config

base_model = tf.keras.applications.Xception(input_shape=[299, 299, 3], include_top=False)
base_model.summary()
base_model.trainable = False
# Use the activations of these layers
layer_names = [
    'add',   # 64x64
    'add_1',   # 64x64
    'add_2',   # 32x32
    'add_3',  # 8x8
    'add_4',      # 4x4
    'add_5'
]
layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

up_stack = [
    ConvBlock(1024, stride=1),   # 32x32 -> 64x64
    ConvBlock(512, stride=1),  # 4x4 -> 8x8
    ConvBlock(256, stride=1),  # 8x8 -> 16x16
    ConvBlock(128, padding='valid', kernel=2, stride=1, upscale=True),  # 16x16 -> 32x32
    ConvBlock(128, stride=1, upscale=True),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[img_size, img_size, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    # x = tf.keras.layers.Dropout(.5)(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='valid')  #64x64 -> 128x128w

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
print(model.output_shape)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[
                  'accuracy',
              ])

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[
                              DisplayCallback(),
                              tf.keras.callbacks.ModelCheckpoint(
                                  './Models/ImageSegX',
                                  save_best_only=True,
                                  monitor='val_accuracy'
                              ),
                          ])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)
