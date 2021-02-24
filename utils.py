import json
import os


def check_checkpoint_config(config):
    if not os.path.exists(config):
        raise FileNotFoundError('Config file does not exist or not specified...')

    config = json.loads(open(config, 'r').read())

    for k in ['num_layers', 'units', 'd_model', 'num_heads', 'dropout', 'max_len', 'mapping']:
        if k not in config:
            raise NotImplementedError(f'{k} missing in config.')

    return config
