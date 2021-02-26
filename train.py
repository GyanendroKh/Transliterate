import json
import os

import click
import tensorflow as tf
from click import echo

from data.data import Dataset
from transformer.transformer import transformer
from utils import check_checkpoint_config


@click.command()
@click.option('--epochs', default=10, type=int, help='Number of Epoch to train.')
@click.option('--batch-size', default=64, type=int, help='Number of batch for training.')
@click.option('--layers', default=5, type=int, help='Number of layers.')
@click.option('--units', default=512, type=int, help='Number of units for the Dense Layers.')
@click.option('--d_model', default=256, type=int, help='Model Dimension.')
@click.option('--heads', default=8, type=int, help='Number of heads.')
@click.option('--dropout', default=0.1, type=float, help='Dropout Rate.')
@click.option('--dataset-dir', default='dataset/train', help='Path to the dataset directory.')
@click.option('--dataset-val-dir', default='dataset/validation', help='Path to the validation dataset directory.')
@click.option('--checkpoint', default='./checkpoints', help='Path to save checkpoints.')
@click.option('--restore-checkpoint', is_flag=True, help='Should restore previous checkpoint.')
@click.option('--epoch-to-save', default=5, type=int, help='For every number of epoch, the checkpoint will be save.')
@click.option('--verbose', default=1, type=int, help='Verbose.')
def main(epochs,
         batch_size,
         layers,
         units,
         d_model,
         heads,
         dropout,
         dataset_dir,
         dataset_val_dir,
         checkpoint,
         restore_checkpoint,
         epoch_to_save,
         verbose):
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
            restore_checkpoint = 0
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

    vocab_size = len(mapping)
    buffer_size = len(train_dataset.data)
    val_buffer_size = 0

    if verbose != 0:
        echo(config)

    dataset = train_dataset.export_as_tf_dataset().cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if os.path.exists(dataset_val_dir):
        try:
            validation_dataset = Dataset(dataset_val_dir, config={
                'mapping': mapping,
                'max_len': max_len
            })

            validation_dataset.create()
            dataset_val = validation_dataset.export_as_tf_dataset()
            val_buffer_size = len(validation_dataset.data)
            dataset_val = dataset_val.cache().shuffle(val_buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        except ValueError as e:
            print(e)
            dataset_val = None
    else:
        dataset_val = None

    print('Training Data Size:', buffer_size)
    print('Validation Data Size:', val_buffer_size)

    model = transformer(
        max_len=max_len,
        vocab_size=vocab_size,
        num_layers=layers,
        units=units,
        d_model=d_model,
        num_heads=heads,
        dropout=dropout
    )

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, max_len - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, m, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = m
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            return {
                'm': self.m,
                'warmup_steps': self.warmup_steps
            }

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, max_len - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    if not restore_checkpoint:
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint, max_to_keep=5)

    if restore_checkpoint and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        echo('Checkpoint restored...')

    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, e, logs=None):
            if ((e + 1) % epoch_to_save) == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'\rSaving checkpoint for epoch {e + 1} at {ckpt_save_path}')

    model.fit(
        dataset,
        epochs=epochs,
        callbacks=[CheckpointCallback()],
        validation_data=dataset_val,
        verbose=2 if verbose > 2 else verbose
    )


if __name__ == '__main__':
    main()
