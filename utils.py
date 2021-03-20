import os
import torch
import config



def load_model(model,load_dir,name):
    '''
    This model will load the parameters saved on load_dir/name.
    
    Args:
        model(torch.nn.Module)
        load_dir(str)
        name(str)
    '''
    file_path=os.path.join(load_dir,name)
    assert os.path.exists(file_path),'file: '+file_path+' doesn\'t exist'
    model.load_state_dict(torch.load(file_path,map_location=config.DEVICE))
    

def save_model(model,save_dir,name):
    '''
    This model's parameters will be saved on save_dir/name.

    Args:
        model(torch.nn.Module)
        save_dir(str)
        name(str)
    '''
    file_path=os.path.join(save_dir,name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if type(model)=='torch.nn.DataParallel':
        torch.save(model.module.state_dict(),file_path)
    else:
        torch.save(model.state_dict(),file_path)


def to_device(data):
    if config.DEVICE.type=='cpu':
        return data.to(config.DEVICE)
    else:
        if issubclass(type(data),torch.nn.Module)==True:
            data=torch.nn.DataParallel(data)
            return data.to(config.DEVICE)
        else:
            return data.to(config.DEVICE)


def variable(func):
    '''
    A decorator. Make a function look like a variable.

    Args:
        func(function)
    '''
    return func()
    
