import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import os.path as path
import config_PL as config
import model
import dataset
import loss
import utils


# TODO: completely revise this
def train(data_loader, net, criterion, net_optim, loss_meter, t, model_name):
    if model_name == 'FANNet':
        for src_img, trgt_label, trgt_img in data_loader:
            batch_num = src_img.size(0)

            output_img = net(src_img, trgt_label)
            train_loss = criterion(output_img, trgt_img)
            loss_meter.update(train_loss.item(), batch_num)
            train_loss.backward()
            net_optim.step()
            t.set_postfix(loss='{:.6f}'.format(loss_meter.value))
            t.update(batch_num)


def val(data_loader, net, loss_meter, t, model_name):
    if model_name == 'FANNet':
        for src_img,trgt_label,trgt_img in data_loader:
            batch_num = src_img.size(0)
            src_img = utils.to_device(src_img)
            trgt_label = utils.to_device(trgt_label)
            trgt_img = utils.to_device(trgt_img)

            with torch.no_grad():
                output_img = net(src_img,trgt_label)
                val_loss = criterion(output_img,trgt_img)

            loss_meter.update(val_loss.item(),batch_num)
            t.set_postfix(loss='{:.6f}'.format(loss_meter.value))
            t.update(batch_num)


if __name__ == '__main__':
    cudnn.benchmark = True

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
        net = utils.to_device(model.FANNet())
    elif config.MODEL == 'ColorNet':
        net = utils.to_device(model.ColorNet())

    # TODO: replace this with the right code
    if config.LOAD == True:
        utils.load_model(net,
                         path.join(config.MODEL_DIR, str(config.EPOCH_START)),
                         config.MODEL+'.pth')

    #train
    best_result=10**10

'''
    for epoch in range(config.EPOCH_START, config.EPOCH):
        with tqdm(total=len(train_data), ncols=80) as t:
            net.train()
            t.set_description('train: {}/{}'.format(epoch+1, config.EPOCH))
            loss_meter.reset()

            train(train_data_loader,
                  net,
                  criterion,
                  net_optim,
                  loss_meter,
                  t,
                  config.MODEL)
            
        
        with tqdm(total=len(val_data), ncols=80) as t:
            net.eval()
            t.set_description('val: {}/{}'.format(epoch+1, config.EPOCH))
            loss_meter.reset()

            val(val_data_loader,
                net,
                loss_meter,
                t,
                config.MODEL)

        if loss_meter.value < best_result:
            best_result = loss_meter.value
            utils.save_model(net,
                             path.join(config.MODEL_DIR, 'best'),
                             config.MODEL+'.pth')

        utils.save_model(net,
                         path.join(config.MODEL_DIR, str(epoch+1)),
                         config.MODEL+'.pth')

'''