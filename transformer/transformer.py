import tensorflow as tf

from .decoder import decoder
from .encoder import encoder


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask


def create_combined_mask(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask


def transformer(max_len, vocab_size, num_layers, units, d_model, num_heads, dropout, training=True, name='transformer'):
    inputs = tf.keras.Input(shape=(max_len,), name='inputs')
    dec_inputs = tf.keras.Input(shape=(max_len - 1,), name='dec_inputs')

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask,
        output_shape=(1, 1, max_len),
        name='trans_enc_pad_mask'
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_combined_mask,
        output_shape=(1, max_len - 1, max_len - 1),
        name='trans_look_ahead_mask'
    )(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask,
        output_shape=(1, 1, max_len),
        name='trans_dec_pad_mask'
    )(inputs)

    enc = encoder(
        seq_len=max_len,
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        training=training
    )
    enc_outputs = enc(inputs=[inputs, enc_padding_mask])

    dec = decoder(
        seq_len=max_len,
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        training=training
    )
    dec_outputs = dec(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

    return tf.keras.Model(
        inputs=[inputs, dec_inputs],
        outputs=outputs,
        name=name
    )
