import numpy as np
from utils.layers import MatMul, SoftmaxWithLoss
from utils.util import preprocess, create_contexts_target, convert_one_hot
from utils import train
from utils.optimizer import *
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"


class simpleCBOW:
    def __init__(self, vocab_size, hidden_size) -> None:
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "The man who can drive himself further once the effort gets painful is the man who will win."
corpus, word2id, id2word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word2id)
contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)

model = simpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = train.Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id2word.items():
    print(word, word_vecs[word_id])