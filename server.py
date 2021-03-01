from concurrent import futures
import grpc
from time import time
import os

from rpc import service_pb2, service_pb2_grpc
from utils import check_checkpoint_config
from transformer.model import Model


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


def serve():
    checkpoint = './checkpoints'
    config = os.path.join(checkpoint, 'config.json')
    config = check_checkpoint_config(config)

    model = Model(config, checkpoint)
    model.restore_checkpoint()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_TransServicer_to_server(Transliterator(model), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server Started...')
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
