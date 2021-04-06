import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

from utils.optimizer import SGD
from data.spiral import load_data
from model.two_layer import TwoLayerNet


class Config:
    EPOCH = 300
    BATCH = 30
    hidden_size = 10
    learning_rate = 1.0


x, t = load_data()
model = TwoLayerNet(2, Config.hidden_size, 3)
optimizer = SGD(Config.learning_rate)

data_size = len(x)
max_iters = data_size // Config.BATCH
total_loss = 0
loss_count = 0
loss_list = []

from utils.train import *

trainer = Trainer(model, optimizer)
trainer.fit(x, t, Config.EPOCH, Config.BATCH, eval_interval=10)
trainer.plot()
exit(0)
