import os

from typing import List
from loader import CorpusLoader
from model import Word2VecModel
from controller import Word2VecController

# 从文件读取语料库
corpus_dir = "corpus"
corpus: List[str] = []
for path in os.listdir(corpus_dir):
    with open(os.path.join(corpus_dir, path), "r") as f:
        corpus.append(f.read())

# 采样训练数据和测试数据
loader = CorpusLoader(corpus)
n_train = 8000
X, Y = loader.sample_data(n=10000)
x_train, y_train = X[:n_train], Y[:n_train]
x_test, y_test = X[n_train:], Y[n_train:]

model = Word2VecModel({
    "vocabulary_size": x_train.shape[1],
    "hidden_layers": [32],
})
controller = Word2VecController(model, {
    "epochs": 1000,
    "batch_size": 1024,
    "shuffle_buffer_size": 100000,
    "save_path": "./kite_word2vec.h5"
})

if __name__ == "__main__":
    arg = os.sys.argv[1]
    if arg == "train":
        controller.train(x_train, y_train, x_test, y_test)
    elif arg == "test":
        controller.test(x_test, y_test)