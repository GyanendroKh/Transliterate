import json
import os
from transformer.model import Model

import click
import tensorflow as tf
from click import echo

from data.data import Dataset
from utils import check_checkpoint_config


@click.command()
@click.option('--epochs', default=10, type=int, help='Number of Epoch to train.')
@click.option('--batch-size', default=64, type=int, help='Number of batch for training.')
@click.option('--layers', default=4, type=int, help='Number of layers.')
@click.option('--units', default=512, type=int, help='Number of units for the Dense Layers.')
@click.option('--d_model', default=128, type=int, help='Model Dimension.')
@click.option('--heads', default=8, type=int, help='Number of heads.')
@click.option('--dropout', default=0.1, type=float, help='Dropout Rate.')
@click.option('--dataset-dir', default='dataset/train', help='Path to the dataset directory.')
@click.option('--checkpoint', default='./checkpoints', help='Path to save checkpoints.')
@click.option('--restore-checkpoint', is_flag=True, help='Should restore previous checkpoint.')
@click.option('--epoch-to-save', default=5, type=int, help='For every number of epoch, the checkpoint will be save.')
@click.option('--verbose', default=1, type=int, help='Verbose.')
def main(
    epochs,
    batch_size,
    layers,
    units,
    d_model,
    heads,
    dropout,
    dataset_dir,
    checkpoint,
    restore_checkpoint,
    epoch_to_save,
    verbose
):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    config_path = os.path.join(checkpoint, 'config.json')
    config = {
        'num_layers': layers,
        'units': units,
        'd_model': d_model,
        'num_heads': heads,
        'dropout': dropout,
    }

    if restore_checkpoint:
        try:
            config = check_checkpoint_config(config_path)
        except FileNotFoundError as e:
            echo(e)
            raise ValueError('Checkpoint config not found.')

        mapping, max_len = config['mapping'], config['max_len']
        train_dataset = Dataset(dataset_dir, config={
            'mapping': mapping,
            'max_len': max_len
        })
        train_dataset.create()
    else:
        train_dataset = Dataset(dataset_dir)
        train_dataset.create()
        config['mapping'], config['max_len'] = train_dataset.mapping, train_dataset.max_len
        mapping, max_len = config['mapping'], config['max_len']

    buffer_size = len(train_dataset.data)

    if verbose != 0:
        echo(config)

    dataset = train_dataset.export_as_tf_dataset()\
        .cache()\
        .shuffle(buffer_size)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    print('Training Data Size:', buffer_size)

    model = Model(config, checkpoint)

    if not restore_checkpoint:
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))
    else:
        model.restore_checkpoint()

    model.train(dataset, epochs, epoch_to_save)


if __name__ == '__main__':
    main()
