import tensorflow as tf

from .attention import MultiHeadAttention
from .encoding import positional_encoding


def decoder_layer(seq_len, units, d_model, num_heads, dropout, training=True, name='decoder_layer'):
    inputs = tf.keras.Input(shape=(seq_len - 1, d_model), name='dec_layer_inputs')
    enc_outputs = tf.keras.Input(shape=(seq_len, d_model), name='dec_layer_enc_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, seq_len - 1, seq_len - 1), name='dec_layer_look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='dec_layer_padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name='attention_1'
    )(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.Dropout(dropout)(attention1, training=training)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name='attention_2'
    )(inputs={
        'query': out1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2, training=training)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + out1)

    ffn_output = tf.keras.layers.Dense(units=units, activation='relu')(out2)
    ffn_output = tf.keras.layers.Dense(units=d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(rate=dropout)(ffn_output, training=training)
    out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + out2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=out3,
        name=name
    )


def decoder(seq_len, vocab_size, num_layers, units, d_model, num_heads, dropout, training=True, name='decoder'):
    inputs = tf.keras.Input(shape=(seq_len - 1,), name='dec_inputs')
    enc_outputs = tf.keras.Input(shape=(seq_len, d_model), name='dec_encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, seq_len - 1, seq_len - 1), name='dec_look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, seq_len), name='dec_padding_mask')

    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += positional_encoding(1000, d_model)[:, :tf.shape(x)[1], :]

    x = tf.keras.layers.Dropout(rate=dropout)(x, training=training)

    for i in range(num_layers):
        x = decoder_layer(
            seq_len=seq_len,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            training=training,
            name='decoder_layer_{}'.format(i),
        )(inputs=[x, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=x,
        name=name
    )
