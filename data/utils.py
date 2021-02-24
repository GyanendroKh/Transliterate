import io
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .constant import Lang, Tokens, known_ext


def read_file(path, num_lines=None):
    return io.open(path).read().strip().split('\n')[:num_lines]


def process_file(file):
    print(f'Processing File {file}.')
    filename = file.split('/')[-1]
    source_lang, target_lang = get_lang_from_filename(filename)
    lines = read_file(file)
    ext = get_file_type(filename)
    bi = is_bidirectional(filename)

    data = []

    for line in tqdm(lines):
        words = line.strip().split('\t' if ext == 'tab' else ',')

        if len(words) != 2:
            continue

        if bi:
            data.append([
                preprocess_word(words[1], start=Tokens.to(source_lang)),
                preprocess_word(words[0])
            ])
        data.append([
            preprocess_word(words[0], start=Tokens.to(target_lang)),
            preprocess_word(words[1])
        ])

    return source_lang, target_lang, data


def preprocess_word(word, start=Tokens.start):
    word = word.lower().strip().replace('\xad', '').replace('\u200d', '')

    return [start, *[c for c in word], Tokens.end]


def encode_word(word, mapping):
    w = []

    for c in word:
        if c in mapping:
            w.append(mapping.index(c))
        else:
            w.append(mapping.index(Tokens.unk))
    return w


def decode_word(word, mapping):
    return [mapping[c] for c in word if mapping[c] != Tokens.pad]


def get_dataset_files(path, known_ext=known_ext):
    files = os.listdir(path)

    files = [
        os.path.join(path, f) for f in os.listdir(path)
        if not os.path.isdir(os.path.join(path, f)) and f.split('.')[-1] in known_ext
    ]

    return files


def to_tf_dataset(data):
    data = np.array(data)

    source = data[:, 0]
    target = data[:, 1]

    return tf.data.Dataset.from_tensor_slices((
        {
            'inputs': source,
            'dec_inputs': target[:, :-1]
        },
        {
            'outputs': target[:, 1:]
        }
    ))


def get_lang_from_filename(name):
    name_without_ext = name.split('.')[0]

    langs = name_without_ext.split('-')

    if len(langs) != 2:
        raise ValueError(f'File {name} has invalid name.')

    source, target = langs[0], langs[1]

    for lang in [source, target]:
        if not Lang.is_supported(lang):
            raise ValueError(f'Lang {lang} is not supported.')

    return source, target


def get_file_type(file):
    ext = file.split('.')[-1]

    if ext not in known_ext:
        raise ValueError(f'Unknown file type {ext} in {file}')

    return ext


def is_bidirectional(file):
    exts = file.split('.')

    return len(exts) == 3 and exts[1] == 'bi'
