import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.utils.data as data
import os
import os.path as path
import config_PL as config
import model_PL as model
import dataset_PL as dataset
import loss
import utils
import pytorch_lightning as pl



# load data
if config.MODEL == 'ColorNet':
    train_data = dataset.ColorNet_train
    val_data = dataset.ColorNet_val

#load model
if config.MODEL == 'FANNet':
    net = model.LightningFANNet()
elif config.MODEL == 'ColorNet':
    net = model.ColorNet()

# TODO: add ability to load from checkpoint
if config.LOAD == True:
    pass

#train
dm = dataset.LightningFANNetDataset()
trainer = pl.Trainer(max_epochs=10, gpus=0, progress_bar_refresh_rate=20)
trainer.fit(net, dm)