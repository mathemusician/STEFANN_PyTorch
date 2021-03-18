import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import os.path as path
import config
import model
import dataset
import loss
import metric
import utils



if __name__=='__main__':
    cudnn.benchmark=True

    #load data
    if config.MODEL=='FANNet':
        train_data=dataset.FANNet_train
        val_data=dataset.FANNet_val
    elif config.MODEL=='ColorNet':
        train_data=dataset.ColorNet_train
        val_data=dataset.ColorNet_val

    train_data_loader=data.DataLoader(dataset=train_data,
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=True)
    val_data_loader=data.DataLoader(dataset=val_data,
                                    batch_size=config.BATCH_SIZE,
                                    shuffle=False)

    #load model
    if config.MODEL=='FANNet':
        net=model.FANNet()
    elif config.MODEL=='ColorNet':
        net=model.ColorNet()

    if config.LOAD==True:
        utils.load_model(net,
                         path.join(config.LOAD_DIR,str(config.EPOCH_START)),
                         config.MODEL+'.pth')

    #set optimizer, loss and meter
    net_optim=optim.Adam(betas=(config.BETAS))
    criterion=loss.MAELoss()
    loss_meter=metric.LossMeter()

    #train
    best_result=10*10

    for epoch in range(config.EPOCH_START,config.EPOCH):
        with tqdm(total=len(train_data),ncol=80) as t:
            net.train()
            t.set_description('train: {}/{}'.format(epoch+1,config.EPOCH))
            loss_meter.reset()

            train(train_data_loader,
                  net,
                  criterion,
                  net_optim,
                  loss_meter,
                  t,
                  config.MODEL)
        
        with tqdm(total=len(val_data),ncol=80) as t:
            net.eval()
            t.set_description('val: {}/{}'.format(epoch+1,config.EPOCH))
            loss_meter.reset()

            val(val_data_loader,
                net,
                loss_meter,
                t,
                config.MODEL)

        if loss_meter.value<best_result
            best_result=loss_meter.value
            utils.save_model(net,
                             path.join(config.MODEL_DIR,str(epoch+1)),
                             config.MODEL+'.pth')

