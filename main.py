import os

from typing import List
from loader import CorpusLoader
from model import Word2VecModel
from workflow import Word2VecWorkflow

if __name__ == "__main__":
    corpus_dir = "corpus"
    corpus: List[str] = []
    for path in os.listdir(corpus_dir):
        with open(os.path.join(corpus_dir, path), "r") as f:
            corpus.append(f.read())
    loader = CorpusLoader(corpus)
    x_train, y_train = loader.sample_data(n=10000)
    x_test, y_test = loader.sample_data(n=2000)
    model = Word2VecModel({
        "vocabulary_size": x_train.shape[1],
        "hidden_layers": [32],
    })
    workflow = Word2VecWorkflow(model, {
        "epochs": 1000,
        "batch_size": 1024,
        "shuffle_buffer_size": 100000,
    })
    workflow.train(x_train, y_train, x_test, y_test)