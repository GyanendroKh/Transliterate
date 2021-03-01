from data.constant import Tokens
from data.utils import decode_word, encode_word, preprocess_word
import tensorflow as tf
import time
from .transformer import Transformer, create_masks


class Model:
    def __init__(self, config, checkpoint):
        self.check_config(config)
        self.config = config
        self.checkpoint = checkpoint

        self.num_layers = config['num_layers']
        self.units = config['units']
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.max_len = config['max_len']
        self.mapping = config['mapping']
        self.vocab_size = len(self.mapping)
        self.pe_input = 1000
        self.pe_target = 1000

        self.model = Transformer(
            self.num_layers,
            self.d_model,
            self.num_heads,
            self.units,
            self.vocab_size,
            self.vocab_size,
            self.pe_input,
            self.pe_target,
            self.dropout
        )

        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint, max_to_keep=5)

    def check_config(self, config):
        for k in ['num_layers', 'units', 'd_model', 'num_heads', 'dropout', 'max_len', 'mapping']:
            if k not in config:
                raise NotImplementedError(f'{k} missing in config.')

    def restore_checkpoint(self):
        if not self.ckpt_manager.latest_checkpoint:
            raise ValueError('Checkpoint not found.')

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

    def train(self, dataset, epochs=10, epoch_to_save=5):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        learning_rate = Model.CustomSchedule(self.d_model)

        optimizers = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions = self.model(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
                loss = self.get_loss_function()(tar_real, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizers.apply_gradients(zip(gradients, self.model.trainable_variables))

            train_loss(loss)
            train_accuracy(self.get_accuracy_function()(tar_real, predictions))

        for epoch in range(epochs):
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for (batch, (inp, tar)) in enumerate(dataset):
                train_step(inp, tar)
                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1}; Batch {batch}; Loss {train_loss.result():.4f}; Accuracy {train_accuracy.result():.4f}')

            if (epoch + 1) % epoch_to_save == 0:
                ckpt_save_path = self.ckpt_manager.save()

                print(f'Saving checkpoint at {ckpt_save_path}')

            print(f'Epoch {epoch + 1}; Loss {train_loss.result():.4f}; Accuracy {train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch is {(time.time() - start):.2f} seconds.\n')

    def infer(self, words, to):
        words = words.split(' ')
        words = [preprocess_word(w, Tokens.to(to)) for w in words]

        results = []

        for word in words:
            if len(word) > self.max_len:
                raise ValueError(f'Length of word greater than max_len. "len({word})" > {self.max_len}')

            inp = encode_word(word, self.mapping)
            inp = tf.expand_dims(inp, axis=0)

            output = tf.expand_dims([self.mapping.index(Tokens.start)], 0)

            for _ in range(self.max_len):
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, output)

                predictions = self.model(inp, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
                predictions = predictions[:, -1:, :]

                predictied_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

                output = tf.concat([output, predictied_id], axis=-1)

                if tf.equal(predictied_id, [self.mapping.index(Tokens.end)]) or len(output[0]) >= self.max_len:
                    break

            decoded = decode_word(tf.squeeze(output, axis=0), self.mapping)[1:-1]
            results.append(''.join(decoded))

        return results

    def get_loss_function(self):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ += mask

            return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

        return loss_function

    def get_accuracy_function(self):
        def accuracy_function(real, pred):
            accuracies = tf.equal(real, tf.argmax(pred, axis=2))

            mask = tf.math.logical_not(tf.math.equal(real, 0))
            accuracies = tf.math.logical_and(mask, accuracies)

            accuracies = tf.cast(accuracies, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)

            return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

        return accuracy_function

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(Model.CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
