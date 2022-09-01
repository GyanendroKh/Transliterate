import grpc
from flask import Flask, request
from flask_cors import CORS

from rpc import service_pb2_grpc, service_pb2

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)


def run(to, sentence):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.TransStub(channel)
        response = stub.infer(service_pb2.SourceWord(word=sentence, to=to))

        return response


@app.route('/api/trans', methods=['GET'])
def trans():
    to = request.args.get('to', '').strip()
    source = request.args.get('source', '').strip()

    if not to or not source:
        return {'error': 'Invalid Request'}, 400

    res = run(to, source)
    print(res)

    return {'output': res.output, 'time': res.time}, 200
