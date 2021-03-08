import grpc
import click
from click import echo

from rpc import service_pb2_grpc, service_pb2


@click.command()
@click.option('--to', '-t', required=True, help='Language to transliterate to.')
@click.argument('sentence')
def run(to, sentence):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.TransStub(channel)
        response = stub.infer(service_pb2.SourceWord(word=sentence, to=to))
    echo(response.output)
    echo(response.time)


if __name__ == '__main__':
    run()
