import numpy as np
from easydict import EasyDict

from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras import Sequential, Model


class Parameter:
    vocabulary_size = 0
    hidden_layers = []

class Word2VecModel(Model):
    def __init__(self, param: dict) -> None:
        super(Word2VecModel, self).__init__()
        self.param = EasyDict(param)
        self._build()

    def _build(self) -> None:
        self.model = Sequential()
        self.model.add(
            Input(
                (self.param.vocabulary_size,),
                name="input_word"))
        for size in self.param.hidden_layers:
            self.model.add(Dense(size))
        self.model.add(Dense(self.param.vocabulary_size))
        self.model.add(Softmax())

    def call(self, x):
        return self.model(x)
