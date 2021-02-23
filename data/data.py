from data.constant import Tokens
import os
from tqdm import tqdm
from .utils import encode_word, get_dataset_files, process_file


class Dataset:
    train_dir = None
    validation_dir = None
    train_data = []
    validation_data = []
    mapping = []
    mapped_train_data = []
    mapped_validation_data = []
    max_len = 0

    def __init__(self, path) -> None:
        self.path = path
        dirs = os.listdir(self.path)

        if 'train' in dirs:
            self.train_dir = os.path.join(self.path, 'train')

        if 'validation' in dirs:
            self.validation_dir = os.path.join(self.path, 'validation')

    def create(self):
        if self.train_dir:
            files = get_dataset_files(self.train_dir)
            for f in files:
                _, _, words = process_file(f)
                self.train_data.extend(words)

        if self.validation_dir:
            files = get_dataset_files(self.validation_dir)
            for f in files:
                _, _, words = process_file(f)
                self.validation_data.extend(words)

        self.__tokenzie()
        self.__create_mapping_and_pad()

    def __tokenzie(self):
        print('\nCreating Tokens.')

        tokens = set()

        for data in [self.train_data, self.validation_data]:
            for word_pair in tqdm(data):
                for word in word_pair:
                    if len(word) > self.max_len:
                        self.max_len = len(word)

                    for char in word:
                        tokens.add(char)

        # Additional Tokens
        self.mapping = [
            Tokens.pad,
            Tokens.unk
        ] + sorted(tokens)

    def __create_mapping_and_pad(self):
        print('\nMapping words to token.')

        for word_pair in tqdm(self.train_data):
            self.mapped_train_data.append([
                encode_word(word_pair[0], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[0]))),
                encode_word(word_pair[1], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[1]))),
            ])

        for word_pair in tqdm(self.validation_data):
            self.mapped_validation_data.append([
                encode_word(word_pair[0], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[0]))),
                encode_word(word_pair[1], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[1]))),
            ])
