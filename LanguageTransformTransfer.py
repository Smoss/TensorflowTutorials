import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

BUFFER_SIZE = 20000
BATCH_SIZE = 256


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2

def encode_self_en(lang):
    lang_inp = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang.numpy()) + [tokenizer_en.vocab_size + 1]
    lang_target = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang_inp, lang_target

def encode_self_pt(lang):
    lang_inp = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang.numpy()) + [tokenizer_pt.vocab_size + 1]
    lang_target = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang_inp, lang_target

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

def tf_encode_en(pt, en):
    result_inp, result_targ = tf.py_function(encode_self_en, [en], [tf.int64, tf.int64])
    result_inp.set_shape([None])
    result_targ.set_shape([None])

    return result_inp, result_targ

def tf_encode_pt(pt, en):
    result_inp, result_targ = tf.py_function(encode_self_pt, [pt], [tf.int64, tf.int64])
    result_inp.set_shape([None])
    result_targ.set_shape([None])

    return result_inp, result_targ


MAX_LENGTH = 40


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_dataset_en = train_examples.map(tf_encode_en, num_parallel_calls=16)
train_dataset_en = train_dataset_en.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset_en = train_dataset_en.cache()
train_dataset_en = train_dataset_en.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset_en = train_dataset_en.prefetch(tf.data.experimental.AUTOTUNE)

train_dataset_pt = train_examples.map(tf_encode_pt, num_parallel_calls=16)
train_dataset_pt = train_dataset_pt.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset_pt = train_dataset_pt.cache()
train_dataset_pt = train_dataset_pt.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset_pt = train_dataset_pt.prefetch(tf.data.experimental.AUTOTUNE)

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode_en)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))
print(pt_batch)
print(en_batch)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)
#
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q,
        k,
        v,
        None
    )
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)



def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

EPOCHS = 30
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

def train(transformer, optimizer, train_dataset, ckpt_manager):
    @tf.function(experimental_relax_shapes=True, input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> portuguese, tar -> english
        for (inp, tar) in tqdm.tqdm(train_dataset, total=BUFFER_SIZE // BATCH_SIZE * 2):
            train_step(inp, tar)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        # if epoch == 0:
        #     print(transformer.layers[0].get_weights())


def evaluate(inp_sentence, transformer, enc_tokenizer, dec_tokenizer):
    start_token = [enc_tokenizer.vocab_size]
    end_token = [enc_tokenizer.vocab_size + 1]
    print(enc_tokenizer.vocab_size)
    print(dec_tokenizer.vocab_size)

    inp_sentence = start_token + enc_tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [dec_tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == dec_tokenizer.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer, enc_tokenizer, dec_tokenizer):
    fig = plt.figure(figsize=(16, 8))

    sentence = enc_tokenizer.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [enc_tokenizer.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([dec_tokenizer.decode([i]) for i in result
                            if i < dec_tokenizer.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, transformer, enc_tokenizer, dec_tokenizer, plot=''):
    result, attention_weights = evaluate(sentence, transformer, enc_tokenizer, dec_tokenizer)

    predicted_sentence = dec_tokenizer.decode([i for i in result
                                              if i < dec_tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot, enc_tokenizer, dec_tokenizer)


num_layers = 8
d_model = 256
dff = 768
num_heads = 8

en_vocab_size = tokenizer_en.vocab_size + 2
pt_vocab_size = tokenizer_pt.vocab_size + 2
dropout_rate = 0.1

def main_func():
    learning_rate_eng = CustomSchedule(d_model)
    learning_rate_pt = CustomSchedule(d_model)
    learning_rate = CustomSchedule(d_model)

    optimizer_eng = tf.keras.optimizers.Adam(learning_rate_eng, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    optimizer_pt = tf.keras.optimizers.Adam(learning_rate_pt, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam(learning_rate_pt, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    transformer_en = Transformer(num_layers, d_model, num_heads, dff,
                              en_vocab_size, en_vocab_size,
                              pe_input=en_vocab_size,
                              pe_target=en_vocab_size,
                              rate=dropout_rate)

    transformer_pt = Transformer(num_layers, d_model, num_heads, dff,
                              pt_vocab_size, pt_vocab_size,
                              pe_input=pt_vocab_size,
                              pe_target=pt_vocab_size,
                              rate=dropout_rate)

    transformer_en_to_pt = Transformer(num_layers, d_model, num_heads, dff,
                              pt_vocab_size, en_vocab_size,
                              pe_input=pt_vocab_size,
                              pe_target=en_vocab_size,
                              rate=dropout_rate)

    transformer_en_to_pt_trained = Transformer(num_layers, d_model, num_heads, dff,
                              pt_vocab_size, en_vocab_size,
                              pe_input=pt_vocab_size,
                              pe_target=en_vocab_size,
                              rate=dropout_rate)
    # print(transformer_en.layers)

    checkpoint_path = "./checkpoints/LanguageTransformer"

    ckpt = tf.train.Checkpoint(
        transformer_eng=transformer_en,
        transformer_pt=transformer_pt,
        transformer_en_to_pt_trained=transformer_en_to_pt_trained,
        optimizer_eng=optimizer_eng,
        optimizer_pt=optimizer_pt,
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # print(transformer_pt.layers[0].get_weights())

    translate("this is a problem we have to solve.", transformer_en, tokenizer_en, tokenizer_en)
    train(transformer_en, optimizer_eng, train_dataset_en, ckpt_manager)
    translate("este é um problema que temos que resolver.", transformer_pt, tokenizer_pt, tokenizer_pt)
    train(transformer_pt, optimizer_pt, train_dataset_pt, ckpt_manager)
    train(transformer_en_to_pt_trained, optimizer, train_dataset, ckpt_manager)
    # print(transformer_en.input_shape)
    # print(transformer_pt.input_shape)

    # transformer_en_to_pt.build(transformer_en.input_shape)
    translate("this is a problem we have to solve.", transformer_en, tokenizer_en, tokenizer_en)
    translate("este é um problema que temos que resolver.", transformer_pt, tokenizer_pt, tokenizer_pt)

    translate("este é um problema que temos que resolver.", transformer_en_to_pt, tokenizer_pt, tokenizer_en)

    transformer_en_to_pt.layers[0].set_weights(transformer_pt.layers[0].get_weights())

    for i in range(1, len(transformer_en_to_pt.layers)):
        transformer_en_to_pt.layers[i].set_weights((transformer_en.layers[i].get_weights()))

    translate("este é um problema que temos que resolver.", transformer_en_to_pt, tokenizer_pt, tokenizer_en)
    translate("este é um problema que temos que resolver.", transformer_en_to_pt_trained, tokenizer_pt, tokenizer_en)
    print("Real translation: this is a problem we have to solve .")
    translate("os meus vizinhos ouviram sobre esta ideia.", transformer_en_to_pt, tokenizer_pt, tokenizer_en)
    translate("os meus vizinhos ouviram sobre esta ideia.", transformer_en_to_pt_trained, tokenizer_pt, tokenizer_en)
    print("Real translation: and my neighboring homes heard about this idea .")
    translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.", transformer_en_to_pt, tokenizer_pt, tokenizer_en)
    translate(
        "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.",
        transformer_en_to_pt_trained, tokenizer_pt, tokenizer_en)
    print(
        "Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")
    translate("este é o primeiro livro que eu fiz.", transformer_en_to_pt, tokenizer_pt, tokenizer_en, plot='decoder_layer4_block2')
    translate("este é o primeiro livro que eu fiz.", transformer_en_to_pt_trained, tokenizer_pt, tokenizer_en,
              plot='decoder_layer4_block2')
    print("Real translation: this is the first book i've ever done.")

if __name__ == "__main__":
    main_func()