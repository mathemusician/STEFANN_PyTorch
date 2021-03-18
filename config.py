import os
import torch



BATCH_SIZE=0
MODEL='FANNet'

if MODEL=='FANNet':
    LR=0.1**3
    MODEL_DIR='./Model/FANNet'
    EPOCH_START=0
    EPOCH=0
    BETAS=(0.9,0.999)
    LAMBDA=0.1**7
elif MODEL=='ColorNet':
    MODEL_DIR='./Model/ColorNet'
    EPOCH_START=0
    EPOCH=0


LOAD=EPOCH_START!=0
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SRC_CHRS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
TRGT_CHRS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

