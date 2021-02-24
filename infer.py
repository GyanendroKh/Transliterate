import os
from time import time

import click
import tensorflow as tf
from click import echo

from data.utils import preprocess_word, encode_word, decode_word
from data.constant import Tokens
from transformer.transformer import transformer
from utils import check_checkpoint_config


def evaluate(config, sentence, to):
    config_keys = ['model', 'mapping', 'max_len']
    for i in config_keys:
        if i not in config:
            raise NotImplementedError(f'{i} not found in config.')

    model = config['model']
    mapping = config['mapping']
    max_len = config['max_len']

    words = sentence.split(' ')
    words = [preprocess_word(w, Tokens.to(to)) for w in words]

    results = []

    for word in words:
        if len(word) > max_len:
            raise TypeError(f'Length of the word should be <= `max_len`\nWord: {word}; max_len: {max_len}')
        inp = encode_word(word, mapping) + ([mapping.index(Tokens.pad)] * (max_len - len(word)))
        inp = tf.expand_dims(inp, axis=0)

        output = tf.expand_dims([mapping.index(Tokens.end)], 0)

        for i in range(max_len):
            output_in = output
            pad = tf.expand_dims([mapping.index(Tokens.pad)] * (max_len - output.shape[1] - 1), 0)
            if not pad.shape[1] == 0:
                output_in = tf.concat([output, pad], axis=-1)

            predictions = model(
                inputs=[inp, output_in],
                training=False
            )

            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if tf.equal(predicted_id, [mapping.index(Tokens.end)]):
                break

            output = tf.concat([output, predicted_id], axis=-1)

            if len(output[0]) >= max_len:
                break

        decoded = decode_word(tf.squeeze(output, axis=0), mapping)[1:]
        results.append(''.join(decoded))

    return results


@click.command()
@click.option('--checkpoint', default='./checkpoints', help='Path to checkpoints to restore model.')
@click.option('--to', '-t', required=True, help='Language to transliterate to.')
@click.argument('sentence')
def main(checkpoint, to, sentence):
    config = os.path.join(checkpoint, 'config.json')
    config = check_checkpoint_config(config)

    num_layers = config['num_layers']
    units = config['units']
    d_model = config['d_model']
    num_heads = config['num_heads']
    dropout = config['dropout']
    max_len = config['max_len']
    mapping = config['mapping']
    vocab_size = len(mapping)

    model = transformer(
        max_len=max_len,
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        echo('Checkpoint restored...')
    else:
        raise FileNotFoundError('No checkpoint found.')

    start = time()
    results = evaluate({
        'model': model,
        'mapping': mapping,
        'max_len': max_len
    }, sentence, to)

    echo(f'Sentence       : {sentence}')
    echo(f'Transliterated : {" ".join(results)}')
    echo(f'Took {(time() - start):.2f}s')


if __name__ == "__main__":
    main()
