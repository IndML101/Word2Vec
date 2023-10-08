import os
import math
import numpy as np
import random
from collections import Counter
import pickle
from typing import List, Any


class DataIO:
    def __init__(self, root_dir:str, filepath: List[str], vocab_size=10000):
        self.root_dir = root_dir
        self.vocab_size = vocab_size
        self.data = list()
        for file in filepath:
            path = os.path.join(root_dir, file)
            with open(path, 'r') as f:
                self.data.extend(f.read().strip().split())

    def get_metadata(self):
        word_counts = Counter(self.data).most_common(self.vocab_size-1)
        word2index = dict()
        
        for i, word in enumerate(word_counts):
            word2index[word[0]] = i

        unk_index = len(word2index)
        word2index['<UNK>'] = unk_index
        index2word = {v:k for k, v in word2index.items()}

        unk_count = 0
        processed_data = list()
        for word in self.data:
            if word in word2index:
                idx = word2index[word]
            else:
                idx = unk_index
                unk_count += 1
            processed_data.append(idx)
        word_counts.append(('<UNK>', unk_count))
        word_counts = [el[1] for el in word_counts]

        return processed_data, word_counts, word2index, index2word

    def save_data(self, data, filepath, filename):
        path = os.path.join(self.root_dir, filepath)
        if os.path.exists(path):
            path = os.path.join(self.root_dir, filepath, filename)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            path = os.path.join(self.root_dir, filepath)
            os.makedirs(path,exist_ok=True)
            path = os.path.join(self.root_dir, filepath, filename)
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def process_data(self, filepath, fnames):
        metadata = self.get_metadata()
        for i, data in enumerate(metadata):
            self.save_data(data, filepath, fnames[i])

    @staticmethod
    def load_data(root_dir, filepath, fnames):
        metadata = list()
        for file in fnames:
            path = os.path.join(root_dir, filepath, file)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    metadata.append(pickle.load(f))
            else:
                raise FileNotFoundError(
                    'No such file or directory: {}'.format(path)
                )

        return metadata

    @staticmethod
    def load_test_data(root_dir,filepath, w2ipath, i2wpath):
        pass


class DataLoader:
    def __init__(self, data, word_counts, word2index, index2word, exp_const):
        self.data = data
        self.data_indices = np.arange(len(data))
        self.word_counts = word_counts
        self.indices = np.arange(len(word_counts))
        self.N = sum(word_counts)
        self.p_word = np.array(list(map(lambda count: math.pow(count, exp_const)/self.N, word_counts)))
        self.p_word /= self.p_word.sum()
        self.word2index = word2index
        self.index2word = index2word


    def get_negative_sample(self, inputs, k):
        samples = list()
        for word in inputs:
            sample = list()
            i = 0
            while i < k:
                wr = np.random.choice(self.indices, p=self.p_word, replace=False)
                if wr != word[0]:
                    i += 1
                    sample.append(wr)
            samples.append(sample)

        return np.array(samples)

    def get_batch(self, window, batch_size):
        outputs = np.zeros(shape=(batch_size, 2*window), dtype=int)
        inputs = np.zeros(shape=(batch_size, 1), dtype=int)
        i = 0
        while i < batch_size:
            try:
                wi = np.random.choice(self.data_indices, size=1)
                idx = wi[0]
                np.append(inputs, [self.data[idx]])
                wo = self.data[idx+1:idx+window+1].extend(self.data[idx-window:idx])
                np.append(outputs, wo)
                i+=1
            except:
                continue
        
        return inputs, outputs

    def get_word(self, idx):
        return self.index2word[idx]

    def get_index(self, word):
        return self.word2index[word]