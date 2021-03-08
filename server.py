from concurrent import futures
import click
import grpc
import signal
from time import time
import os
from utils import check_checkpoint_config
from rpc import service_pb2, service_pb2_grpc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

from transformer.model import Model  # noqa


class Transliterator(service_pb2_grpc.TransServicer):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def infer(self, request, context):
        start = time()
        output = self.model.infer(request.word, request.to)
        end = time() - start
        output = ' '.join(output)
        return service_pb2.Output(output=output, time=end)


@click.command()
@click.option('--checkpoint', default='./checkpoints', help='Path to checkpoints to restore model.')
def serve(checkpoint):
    config = os.path.join(checkpoint, 'config.json')
    config = check_checkpoint_config(config)

    model = Model(config, checkpoint)
    model.restore_checkpoint()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_TransServicer_to_server(Transliterator(model), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server Started...')

    def on_done(signum, frame):
        print()
        print('Stopping Server.')
        server.stop(None)

    signal.signal(signal.SIGINT, on_done)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
