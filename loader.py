from typing import Dict, List, Set, Tuple
import numpy as np

from nltk.tokenize import TreebankWordTokenizer


class CorpusLoader:
    def __init__(self, corpus: List[str]) -> None:
        self.corpus = corpus
        self.data: List[np.ndarray] = []
        self._to_one_hot()

    def _to_one_hot(self) -> None:
        '''
        将语料库中的文档进行分词，并编码为one-hot
        '''
        self.data.clear()
        self.indices_map: Dict[str, int] = {}
        self.inv_map: Dict[int, str] = {}
        word_set: Set[str] = set([])
        tokens_list: List[List[str]] = []
        tokenizer = TreebankWordTokenizer()

        # 获取所有单词
        for doc in self.corpus:
            tokens = tokenizer.tokenize(doc)
            tokens_list.append(tokens)
            word_set = word_set.union(set(tokens))

        # 建立 单词:单词编号 的字典
        i = 0
        for word in word_set:
            self.indices_map[word] = i
            self.inv_map[i] = word
            i += 1

        # 转换为one-hot编码
        for tokens in tokens_list:
            self.data.append(self.words2onehots(tokens))

    def _sample_one_data(self, win_size: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        @param win_size: 窗口大小

        @return: Tuple[np.ndarray, np.ndarray] (输入, 输出), 输入是周围向量的和
        '''
        doc: np.ndarray = np.random.choice(self.data)
        s: int = np.random.randint(doc.shape[0])
        pivot: int = np.clip(
            np.random.randint(s, s + win_size), 0, doc.shape[0] - 1)
        inputs: np.ndarray = np.sum(
            np.concatenate(
                (doc[s:pivot], doc[pivot + 1:s + win_size]), axis=0),axis=0)
        output: np.ndarray = doc[pivot]
        return inputs, output

    def sample_data(self, win_size=4, n=32) -> Tuple[np.ndarray, np.ndarray]:
        '''
        @param win_size: 窗口大小

        @param batch_size: batch大小

        @return: Tuple[np.ndarray, np.ndarray] (输入, 输出), 输入输出的第一维大小均为(win_size-1)*batch_size
        '''
        X: List[np.ndarray] = []
        Y: List[np.ndarray] = []
        for _ in range(n):
            x, y = self._sample_one_data(win_size)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def onehots2words(self, onehots: np.ndarray) -> List[str]:
        numeric = np.argmax(onehots, axis=-1)
        return [self.inv_map[num] for num in numeric]

    def words2onehots(self, words: List[str]) -> np.ndarray:
        numeric = [self.indices_map[word] for word in words]
        one_hots = np.eye(len(self.indices_map))
        return one_hots[numeric]
