import tensorflow as tf

from .attention import MultiHeadAttention
from .encoding import positional_encoding


def encoder_layer(seq_len, units, d_model, num_heads, dropout, name='encoder_layer'):
    inputs = tf.keras.Input(shape=(seq_len, d_model), name='enc_layer_inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='enc_layer_padding_mask')

    attention = MultiHeadAttention(
        d_model, num_heads, name='attention'
    )({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        outputs=outputs,
        name=name
    )


def encoder(seq_len, vocab_size, num_layers, units, d_model, num_heads, dropout, name='encoder'):
    inputs = tf.keras.Input(shape=(seq_len,), name='enc_inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='enc_padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings += positional_encoding(vocab_size, d_model)[:, :tf.shape(embeddings)[1], :]

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            seq_len=seq_len,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='encoder_layer_{}'.format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        outputs=outputs,
        name=name
    )
