import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.utils.data as data
import os
import os.path as path
import config_PL as config
import model
import dataset
import loss
import utils



#load data
if config.MODEL == 'FANNet':
    train_data = dataset.FANNet_train
    val_data = dataset.FANNet_val
elif config.MODEL == 'ColorNet':
    train_data = dataset.ColorNet_train
    val_data = dataset.ColorNet_val


train_data_loader = data.DataLoader(dataset=train_data,
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=8)

val_data_loader = data.DataLoader(dataset=val_data,
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=8)


#load model
if config.MODEL == 'FANNet':
    net = model.FANNet()
elif config.MODEL == 'ColorNet':
    net = model.ColorNet()

# TODO: add ability to load from checkpoint
if config.LOAD == True:
    pass

#train
trainer = pl.Trainer(max_epochs=1, gpus=None, progress_bar_refresh_rate=1)
trainer.fit(net)