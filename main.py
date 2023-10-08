import os, pprint as pp
import torch
from src.utils import DataIO, DataLoader
from src.models import Word2Vec, EvaluateSGNS
import gc


ROOTDIR = os.path.dirname(__file__)
FILEPATH = 'Artifacts/metadata'
FNAMES = [
    'processed_data.pkl',
    'word_counts.pkl',
    'word2index.pkl',
    'index2word.pkl'
]
datapath = ['Data/ds2.txt']
MODELPATH = 'Artifacts/model/word2vec-2023-10-07 22:02:30.387717.pt'
SOURCEFILE = ['Data/ds1.txt']
W2I = 'Artifacts/metadata/word2index.pkl'
I2W = 'Artifacts/metadata/index2word.pkl'

# data_io = DataIO(ROOTDIR, filepath=datapath, vocab_size=10000)
# data_io.process_data(filepath=FILEPATH, fnames=FNAMES)

# metadata = DataIO.load_data(root_dir=ROOTDIR, filepath=FILEPATH, fnames=FNAMES)
# loader = DataLoader(*metadata, exp_const=3/4)

# inputs, outputs = loader.get_batch(window=5, batch_size=32)
# print(inputs.shape)
# print(outputs.shape)

# negative_samples = loader.get_negative_sample(inputs=inputs, k=20)
# print(negative_samples.shape)

# torch.cuda.empty_cache()
# gc.collect()

# word2vec = Word2Vec(
#     root_dir=ROOTDIR,
#     train=True,
#     sgd=False,
#     process_data=False,
#     source_filepath=[],
#     metadata_filepath=FILEPATH,
#     metadata_fnames=FNAMES,
#     vocab_size=10000,
#     embedding_dim=300
# )

# loss, modelname = word2vec.train(epochs=10, steps=10000, window=4, k=10, batch_size=128, loss_history=100)


# model = Word2Vec.load_sgns(vocab_size=10000, embedding_dim=300, path=MODELPATH)
# evaluator = EvaluateSGNS(
#     model=model,
#     root_dir=ROOTDIR,
#     source_filepath=SOURCEFILE,
#     w2i_path=W2I,
#     i2w_path=I2W
# )

# output = evaluator.evaluate(ksamples=10, top_k=5)
# pp.pprint(output)

if __name__ == '__main__':
    pass