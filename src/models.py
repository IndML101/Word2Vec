import os, pickle
from collections import Counter
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch import nn
from src.utils import DataIO, DataLoader
from torch.optim import SGD, Adam
from collections import deque

class SGNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs, outputs, negative_sample):
        w_i = self.input_embedding(inputs)
        w_o = self.output_embedding(outputs)
        w_r = self.output_embedding(negative_sample)

        w_o = w_o.transpose(1,-1)
        w_r = w_r.transpose(1,-1)
        maximize_loss = torch.bmm(w_i, w_o).sigmoid().log().squeeze()
        minimize_loss = torch.bmm(w_i.negative(), w_r).sigmoid().log().sum(-1)

        return -(torch.add(maximize_loss, minimize_loss)).mean()

    def predict(self, inputs):
        return self.input_embedding(inputs)


class Word2Vec:
    def __init__(
        self,
        root_dir,
        train=True,
        sgd=True,
        process_data=False,
        source_filepath=[],
        metadata_filepath=None,
        metadata_fnames=[],
        vocab_size=10000,
        embedding_dim=50,
        exp_const=3/4,
        learning_rate=1e-5
    ):
        self.root_dir = root_dir
        if process_data:
            data_io = DataIO(root_dir, source_filepath, vocab_size)
            data, word_counts, word2index, index2word = data_io.get_metadata()
        else:
            data, word_counts, word2index, index2word = DataIO.load_data(
                root_dir,
                metadata_filepath,
                metadata_fnames
            )
        self.data_loader = DataLoader(
            data,
            word_counts,
            word2index,
            index2word,
            exp_const
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = SGNS(vocab_size, embedding_dim).to(self.device)
        if sgd:
            self.optimizer = SGD(self.model.parameters(), learning_rate)
        else:
            self.optimizer = Adam(self.model.parameters(), learning_rate)

    def init_weights(self, model):
        for name, param in model.named_parameters():
            nn.init.xavier_normal_(param.data, 10)
    
    def train(
        self,
        epochs=20,
        steps=10000,
        batch_size=64,
        window=5,
        k=20,
        loss_history=500,
        output_dir='Artifacts/model'
    ):
        path = os.path.join(self.root_dir, output_dir)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        self.model.apply(self.init_weights)
        totalsteps = steps*epochs
        running_loss = deque(maxlen=loss_history)
        epoch_loss = []
        step_no = 0
        for epoch in range(epochs):
            for step in range(steps):
                inputs, outputs = self.data_loader.get_batch(window, batch_size)
                samples = self.data_loader.get_negative_sample(inputs, k)

                inputs = torch.tensor(inputs, dtype=torch.long).to(self.device)
                outputs = torch.tensor(outputs, dtype=torch.long).to(self.device)
                samples = torch.tensor(samples, dtype=torch.long).to(self.device)

                loss = self.model(inputs, outputs, samples)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss.append(round(loss.item(), 5))
                
                if not step % loss_history:
                    mean_step_loss = np.mean(running_loss)
                    epoch_loss.append(mean_step_loss)
                    step_no += loss_history
                    completion = round(step_no/totalsteps*100, 2)
                    print(f"loss at step no. {step_no} of {totalsteps} steps is {mean_step_loss}, job is {completion}% complete", end='\r')

        modelname = 'word2vec-{}.pt'.format(dt.now())
        torch.save(self.model.state_dict(), os.path.join(path, modelname))
        
        return epoch_loss, modelname

    def plot_loss(self, epoch_loss, modelname, loss_history, path='Artifacts/plots'):
        fname = 'Loss history ' + ''.join(modelname.split('.')[:-1])
        path = os.path.join(self.root_dir, path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        fig = plt.figure(figsize=[10,7])
        iter_key = 'Iterations x {}'.format(loss_history)
        loss_data = {
            'Loss':epoch_loss,
            iter_key: list(range(1, len(epoch_loss)+1)) 
        }
        sns.lineplot(data=loss_data, x=iter_key, y='Loss')
        plt.savefig(os.path.join(path, fname))
        # plt.show();

    def plot_model_parameters(self):
        self.model.parameters()
    
    @staticmethod
    def load_sgns(vocab_size, embedding_dim, path):
        model = SGNS(vocab_size, embedding_dim)
        model.load_state_dict(torch.load(path))
        return model

class EvaluateSGNS:
    def __init__(
        self,
        model,
        root_dir,
        source_filepath=[],
        w2i_path='',
        i2w_path=''
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.root_dir = root_dir
        
        self.data = list()
        for file in source_filepath:
            path = os.path.join(root_dir, file)
            with open(path, 'r') as f:
                self.data.append(f.read().strip().split())

        with open(os.path.join(root_dir, w2i_path), 'rb') as f:
            self.word2index = pickle.load(f)

        with open(os.path.join(root_dir, i2w_path), 'rb') as f:
            self.index2word = pickle.load(f)

    def load_evaluation_data(self):
        data_list = list()
        word_counts_list = list()

        for data in self.data:
            word_counts = Counter(data)
            word_counts_filtered = dict()
            encoded_data = list()
            unk_index = self.word2index.get('<UNK>', len(self.word2index))

            for word in data:
                idx = self.word2index.get(word, unk_index)
                encoded_data.append(idx)

                if idx != unk_index:
                    word_counts_filtered[word] = word_counts[word]
                else:
                    word_counts_filtered['<UNK>'] = word_counts_filtered.get('<UNK>', 0) + word_counts[word]

            data_list.append(encoded_data)
            word_counts_list.append(word_counts_filtered)

        return data_list, word_counts_list

    def choose_data_set(self, n_data_points):
        return np.random.choice(range(n_data_points))

    def sample_words(self, data, word_counts, ksamples=20):
        indices = len(word_counts)
        samples = list()
        i = 0
        while i < ksamples:
            wr = np.random.choice(indices, replace=False)
            i += 1
            samples.append(wr)

        return np.array(samples)

    def get_most_similar_words(self, word, topk=5):
        idx = self.word2index[word]
        idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(self.device)
        embedding = self.model.predict(idx)
        similarity = torch.mm(embedding, self.model.input_embedding.weight.transpose(0, 1))
        most_similar = torch.topk(similarity.unsqueeze(0), topk+1).indices
        most_similar = most_similar.squeeze()
        word_list = []
        for k in range(1, topk+1):
            similar_word = self.index2word[most_similar[k].item()]
            word_list.append(similar_word)
        return word_list

    def evaluate(self, ksamples=20, top_k=5):
        data_list, word_counts_list = self.load_evaluation_data()
        data_idx = self.choose_data_set(len(data_list))
        data, word_counts = data_list[data_idx], word_counts_list[data_idx]

        sample_words = self.sample_words(data, word_counts, ksamples)
        sample_words = [self.index2word[idx] for idx in sample_words]

        return {word:self.get_most_similar_words(word, top_k) for word in sample_words}