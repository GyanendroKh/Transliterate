from data.constant import Tokens
import os
from .utils import is_bidirectional, read_file, get_lang_from_filename, get_dataset_files, get_file_type, preprocess_word


class Dataset:
    train_dir = None
    validation_dir = None
    train_data = []
    validation_data = []

    def __init__(self, path) -> None:
        self.path = path
        self.init()

    def init(self):
        dirs = os.listdir(self.path)

        if 'train' in dirs:
            self.train_dir = os.path.join(self.path, 'train')

        if 'validation' in dirs:
            self.validation_dir = os.path.join(self.path, 'validation')

    def create(self):
        if self.train_dir:
            files = get_dataset_files(self.train_dir)
            for f in files:
                _, _, words = self.__process_file(f)
                self.train_data.extend(words)

        if self.validation_dir:
            files = get_dataset_files(self.validation_dir)
            for f in files:
                _, _, words = self.__process_file(f)
                self.validation_data.extend(words)

    def __process_file(self, file):
        filename = file.split('/')[-1]
        source_lang, target_lang = get_lang_from_filename(filename)
        lines = read_file(file)
        ext = get_file_type(filename)
        bi = is_bidirectional(filename)

        data = []

        for line in lines:
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
