import os
from time import time
from transformer.model import Model

import click
from click import echo

from utils import check_checkpoint_config


@click.command()
@click.option('--checkpoint', default='./checkpoints', help='Path to checkpoints to restore model.')
@click.option('--to', '-t', required=True, help='Language to transliterate to.')
@click.argument('sentence')
def main(checkpoint, to, sentence):
    config = os.path.join(checkpoint, 'config.json')
    config = check_checkpoint_config(config)

    model = Model(config, checkpoint)
    model.restore_checkpoint()

    start = time()
    results = model.infer(sentence, to)
    end = time() - start

    echo(f'Sentence       : {sentence}')
    echo(f'Transliterated : {" ".join(results)}')
    echo(f'Took {(end):.2f}s')


if __name__ == "__main__":
    main()
