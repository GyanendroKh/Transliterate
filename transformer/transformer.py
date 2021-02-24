import tensorflow as tf

from .decoder import decoder
from .encoder import encoder


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


def transformer(max_len, vocab_size, num_layers, units, d_model, num_heads, dropout, name='transformer'):
    inputs = tf.keras.Input(shape=(max_len,), name='inputs')
    dec_inputs = tf.keras.Input(shape=(max_len - 1,), name='dec_inputs')

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask,
        output_shape=(1, 1, max_len),
        name='trans_enc_pad_mask'
    )(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
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
    )
    dec_outputs = dec(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

    return tf.keras.Model(
        inputs=[inputs, dec_inputs],
        outputs=outputs,
        name=name
    )
