from tqdm import tqdm
from .utils import encode_word, get_dataset_files, process_file, to_tf_dataset
from data.constant import Tokens


class Dataset:
    data_dir = None
    data = []
    mapping = []
    mapped_data = []
    max_len = 0

    def __init__(self, path) -> None:
        self.data_dir = path

    def create(self):
        if self.data_dir:
            files = get_dataset_files(self.data_dir)
            for f in files:
                _, _, words = process_file(f)
                self.data.extend(words)

        if len(self.data) == 0:
            return

        self.__tokenize()
        self.__create_mapping_and_pad()

    def export_as_tf_dataset(self):
        if len(self.mapped_data) == 0:
            raise ValueError('There is no mapped data.')

        train_dataset = to_tf_dataset(self.mapped_data)

        return train_dataset

    def __tokenize(self):
        print('\nCreating Tokens.')

        tokens = set()

        for word_pair in tqdm(self.data):
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

        for word_pair in tqdm(self.data):
            self.mapped_data.append([
                encode_word(word_pair[0], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[0]))),
                encode_word(word_pair[1], self.mapping) + ([self.mapping.index(Tokens.pad)] * (self.max_len - len(word_pair[1]))),
            ])
