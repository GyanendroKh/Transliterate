import os
from time import time
from utils import check_checkpoint_config
from flask import Flask, request
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

from transformer.model import Model  # noqa

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

checkpoint = './checkpoints'

config = os.path.join(checkpoint, 'config.json')
config = check_checkpoint_config(config)

model = Model(config, checkpoint)
model.restore_checkpoint()


def run(to, sentence):
    start = time()
    output = model.infer(sentence, to)
    end = time() - start
    output = ' '.join(output)

    return {
        'output': output,
        'time': end
    }


@app.route('/api/trans', methods=['GET'])
def trans():
    to = request.args.get('to', '').strip()
    source = request.args.get('source', '').strip()

    if not to or not source:
        return {'error': 'Invalid Request'}, 400

    res = run(to, source)

    return res, 200
