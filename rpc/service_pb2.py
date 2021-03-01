# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='service.proto',
    package='',
    syntax='proto2',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\rservice.proto\"&\n\nSourceWord\x12\x0c\n\x04word\x18\x01 \x02(\t\x12\n\n\x02to\x18\x02 \x02(\t\"&\n\x06Output\x12\x0e\n\x06output\x18\x01 \x02(\t\x12\x0c\n\x04time\x18\x02 \x02(\x02\x32(\n\x05Trans\x12\x1f\n\x05infer\x12\x0b.SourceWord\x1a\x07.Output\"\x00'
)


_SOURCEWORD = _descriptor.Descriptor(
    name='SourceWord',
    full_name='SourceWord',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='word', full_name='SourceWord.word', index=0,
            number=1, type=9, cpp_type=9, label=2,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='to', full_name='SourceWord.to', index=1,
            number=2, type=9, cpp_type=9, label=2,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=17,
    serialized_end=55,
)


_OUTPUT = _descriptor.Descriptor(
    name='Output',
    full_name='Output',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='output', full_name='Output.output', index=0,
            number=1, type=9, cpp_type=9, label=2,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='time', full_name='Output.time', index=1,
            number=2, type=2, cpp_type=6, label=2,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=57,
    serialized_end=95,
)

DESCRIPTOR.message_types_by_name['SourceWord'] = _SOURCEWORD
DESCRIPTOR.message_types_by_name['Output'] = _OUTPUT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SourceWord = _reflection.GeneratedProtocolMessageType('SourceWord', (_message.Message,), {
    'DESCRIPTOR': _SOURCEWORD,
    '__module__': 'service_pb2'
    # @@protoc_insertion_point(class_scope:SourceWord)
})
_sym_db.RegisterMessage(SourceWord)

Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), {
    'DESCRIPTOR': _OUTPUT,
    '__module__': 'service_pb2'
    # @@protoc_insertion_point(class_scope:Output)
})
_sym_db.RegisterMessage(Output)


_TRANS = _descriptor.ServiceDescriptor(
    name='Trans',
    full_name='Trans',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=97,
    serialized_end=137,
    methods=[
        _descriptor.MethodDescriptor(
            name='infer',
            full_name='Trans.infer',
            index=0,
            containing_service=None,
            input_type=_SOURCEWORD,
            output_type=_OUTPUT,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_TRANS)

DESCRIPTOR.services_by_name['Trans'] = _TRANS

# @@protoc_insertion_point(module_scope)
