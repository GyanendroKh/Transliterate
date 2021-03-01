import grpc

from rpc import service_pb2_grpc, service_pb2


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.TransStub(channel)
        response = stub.infer(service_pb2.SourceWord(word='Helloo', to='mm'))
    print(response.output)
    print(response.time)


if __name__ == '__main__':
    run()
