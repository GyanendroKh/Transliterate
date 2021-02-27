import tensorflow as tf

from .attention import MultiHeadAttention, point_wise_feed_forward_network
from .encoding import positional_encoding


def encoder_layer(seq_len, units, d_model, num_heads, dropout, training=True, name='encoder_layer'):
    inputs = tf.keras.Input(shape=(seq_len, d_model), name='enc_layer_inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='enc_layer_padding_mask')

    mha = MultiHeadAttention(d_model, num_heads, name='attention')
    ffn = point_wise_feed_forward_network(d_model, units)

    layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    dropout1 = tf.keras.layers.Dropout(dropout)
    dropout2 = tf.keras.layers.Dropout(dropout)

    attn_output = mha({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attn_output = dropout1(attn_output, training=training)
    out1 = layer_norm1(inputs + attn_output)

    ffn_output = ffn(out1)
    ffn_output = dropout2(ffn_output, training=training)

    out2 = layer_norm2(out1 + ffn_output)

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        outputs=out2,
        name=name
    )


def encoder(seq_len, vocab_size, num_layers, units, d_model, num_heads, dropout, training=True, name='encoder'):
    inputs = tf.keras.Input(shape=(seq_len,), name='enc_inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='enc_padding_mask')

    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += positional_encoding(1000, d_model)[:, :tf.shape(x)[1], :]

    x = tf.keras.layers.Dropout(rate=dropout)(x, training=training)

    for i in range(num_layers):
        x = encoder_layer(
            seq_len=seq_len,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            training=training,
            name='encoder_layer_{}'.format(i),
        )([x, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask],
        outputs=x,
        name=name
    )
