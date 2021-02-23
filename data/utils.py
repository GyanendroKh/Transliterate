import io
import os
from .constant import Lang, Tokens, known_ext


def read_file(path, num_lines=None):
    return io.open(path).read().strip().split('\n')[:num_lines]


def preprocess_word(word, start=Tokens.start):
    word = word.lower().strip()

    return [start, *[c for c in word], Tokens.end]


def get_dataset_files(path, known_ext=known_ext):
    files = os.listdir(path)

    files = [
        os.path.join(path, f) for f in os.listdir(path)
        if not os.path.isdir(os.path.join(path, f)) and f.split('.')[-1] in known_ext
    ]

    return files


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
