import os
import torch


# Assumption 1
# The fonts start with the same character 
# Assumption 2
# The fonts are numbered the same


BATCH_SIZE=64
MODEL='FANNet'

if MODEL=='FANNet':
    LR=0.1**3
    MODEL_DIR='./Model/FANNet'
    EPOCH_START=0
    EPOCH=64
    BETAS=(0.9,0.999)
    LAMBDA=0.1**7
elif MODEL=='ColorNet':
    LR=0.1**3
    MODEL_DIR='./Model/ColorNet'
    EPOCH_START=0
    EPOCH=0
    BETAS=(0.9,0.999)
    LAMBDA=0.1**7

LOAD = EPOCH_START != 0
SRC_CHRS = '0123456789______ABCDEFGHIJKLMNOPQRSTUVWXYZ______abcdefghijklmnopqrstuvwxyz'
TRGT_CHRS = SRC_CHRS
