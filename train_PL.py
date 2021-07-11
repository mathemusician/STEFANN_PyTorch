#%%
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
from getpaths import getpath
import pandas as pd
from random import randint, sample, seed
import sys




NUM_TRAIN = 3
NUM_VAL = 3
SEED = 3

# make deterministic for reproducibility
seed(SEED)


# make the csv for the dataloader
cwd = getpath()
output_folder = cwd/'Data'/'STEFANN'

# make sure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def pick_random_url(folder):
    # returns getpath url
    files_and_folders = folder.ls()
    # remove DS_Store
    if '.DS_Store' in files_and_folders:
        files_and_folders.remove('.DS_Store')

    random_leaf = sample(files_and_folders, 1)[0]
    return folder/random_leaf

def get_name_int(url):
    name =  url.split(os.sep)[-1]
    name_int = int(name.split('.')[0])
    return name_int

def make_csv(input_folder, output_file_path, num_data_points):
    item_list = []

    # random list of fonts
    for i in range(num_data_points):
        font_url = pick_random_url(input_folder)
        input_image_url = pick_random_url(font_url)
        output_image_url = pick_random_url(font_url)
        output_int = get_name_int(output_image_url)
        item_list.append([input_image_url, output_int, output_image_url])

    # save dataframe as csv
    pd.DataFrame(item_list).to_csv(output_file_path, index=False)

# train csv
train_folder = cwd/'fannet'/'train'
output_file_path = output_folder/'fannet_train.csv'
make_csv(train_folder, output_file_path, NUM_TRAIN)

# val csv
val_folder = cwd/'fannet'/'valid'
output_file_path = output_folder/'fannet_val.csv'
make_csv(val_folder, output_file_path, NUM_VAL)


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
trainer = pl.Trainer(max_epochs=1, gpus=0, progress_bar_refresh_rate=20, check_val_every_n_epoch=2)
trainer.fit(net, dm)

#%%
a = trainer.test()


# %%
