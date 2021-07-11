#%%
import config
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import config_PL as config
import utils
import pytorch_lightning as pl
import torchvision.transforms as transforms
from getpaths import getpath
import os
import matplotlib.pyplot as plt



LEN_TRGT_CHRS = len(config.TRGT_CHRS)

class FANNetLabel(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, label):
        onehot=torch.FloatTensor([0]*LEN_TRGT_CHRS)
        onehot[label] = 1
        return onehot


class FANNetDataset(data.Dataset):
    def __init__(self, csv_file,
                 src_img_trans=None,
                 trgt_label_trans=None,
                 trgt_img_trans=None,
                 lowest_number=None):
        super().__init__()

        self.df = pd.read_csv(csv_file)
        self.src_img_trans = src_img_trans
        self.trgt_label_trans = trgt_label_trans
        self.trgt_img_trans = trgt_img_trans
        self.lowest_number = lowest_number


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        if torch.is_tensor(index) == True:
            index = index.tolist()
            
        info = self.df.loc[index,:]
        src_img_path = info[0]
        trgt_label = int(info[1]) - self.lowest_number - 1
        trgt_img_path = info[2]
        src_img = Image.open(src_img_path).convert('L')
        trgt_img = Image.open(trgt_img_path).convert('L')

        if self.src_img_trans != None:
            src_img = self.src_img_trans(src_img)

        if self.trgt_label_trans != None:
            trgt_label = self.trgt_label_trans(trgt_label)

        if self.trgt_img_trans != None:
            trgt_img = self.trgt_img_trans(trgt_img)

        return src_img, trgt_label, trgt_img


class LightningFANNetDataset(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.trans_src_img = transforms.Compose([transforms.ToTensor()])
        self.trans_trgt_img = transforms.Compose([transforms.ToTensor()])
        self.trans_trgt_label = transforms.Compose([FANNetLabel()])
        self.lowest_number = None
        self.highest_number = None

    def prepare_data(self):
        try:
            #TODO: downloading data for google colab here
            #TODO: maybe make CSV files here
            pass
        finally:
            pass

    def setup(self, stage=None):
        cwd = getpath()
        train_folder = cwd/'fannet'/'train'

        # this is so that one-hot encoded data can start at 0
        for folder_name in train_folder.ls():
            if folder_name == '.DS_Store':
                continue

            file_names = os.listdir(train_folder/folder_name)

            if '.DS_Store' in file_names:
                file_names.remove('.DS_Store')

            file_ints = [int(i.split('.')[0]) for i in file_names]
            file_ints.sort()
            lowest_file_int = file_ints[0]
            highest_file_int = file_ints[-1]

            if self.lowest_number == None:
                self.lowest_number = lowest_file_int
            else:
                if lowest_file_int < self.lowest_number:
                    self.lowest_number = lowest_file_int
            
            if self.highest_number == None:
                self.highest_number = highest_file_int
            else:
                if highest_file_int > self.highest_number:
                    self.highest_number = highest_file_int

        # number of target characters should match the number of unique font image labels
        assert self.highest_number - self.lowest_number == LEN_TRGT_CHRS


    def train_dataloader(self):
        # return 
        fann = FANNetDataset(csv_file='./Data/STEFANN/fannet_train.csv',
                            src_img_trans=self.trans_src_img,
                            trgt_label_trans=self.trans_trgt_label,
                            trgt_img_trans=self.trans_trgt_img,
                            lowest_number=self.lowest_number)

        return data.DataLoader(dataset=[fann[i] for i in range(len(fann))],
                               batch_size=config.BATCH_SIZE,
                               shuffle=True)


    def val_dataloader(self):
        fann = FANNetDataset(csv_file='./Data/STEFANN/fannet_val.csv',
                            src_img_trans=self.trans_src_img,
                            trgt_label_trans=self.trans_trgt_label,
                            trgt_img_trans=self.trans_trgt_img,
                            lowest_number=self.lowest_number)

        return data.DataLoader(dataset=[fann[i] for i in range(len(fann))],
                               batch_size=config.BATCH_SIZE,
                               shuffle=False)


    def test_dataloader(self):
        fann = FANNetDataset(csv_file='./Data/STEFANN/fannet_val.csv',
                            src_img_trans=self.trans_src_img,
                            trgt_label_trans=self.trans_trgt_label,
                            trgt_img_trans=self.trans_trgt_img,
                            lowest_number=self.lowest_number)

        return data.DataLoader(dataset=[fann[0]],
                               batch_size=1,
                               shuffle=False)











trans_input_color = transforms.Compose([transforms.ToTensor()])
trans_input_mask = transforms.Compose([transforms.ToTensor()])
trans_output_color = transforms.Compose([transforms.ToTensor()])

class ColorNetDataset(pl.LightningDataModule):
    def __init__(self,csv_file,
                 input_color_trans=None,
                 input_mask_trans=None,
                 output_color_trans=None):
        super().__init__()

        self.df = pd.read_csv(csv_file)
        self.input_color_trans = input_color_trans
        self.input_mask_trans = input_mask_trans
        self.output_color_trans = output_color_trans


    def __len__(self):
        return len(self.df)


    def __getitem__(self,index):
        if torch.is_tensor(index) == True:
            index = index.tolist()

        info = self.df.loc[index,:]
        input_color_img_path = info[0]
        input_mask_img_path = info[1]
        output_color_img_path = info[2]
        input_color_img = Image.open(input_color_img_path).convert('RGB')
        input_mask_img = Image.open(input_mask_img_path).convert('L')
        output_color_img = Image.open(output_color_img_path).convert('RGB')

        if self.input_color_trans != None:
            input_color_img = self.input_color_trans(input_color_img)

        if self.input_mask_trans != None:
            input_mask_img=self.input_mask_trans(input_mask_img)

        if self.output_color_trans != None:
            output_color_img = self.output_color_trans(output_color_img)

        return input_color_img, input_mask_img, output_color_img


'''
@utils.variable
def ColorNet_train():
    return ColorNetDataset(csv_file='./Data/STEFANN/colornet_train.csv',
                           input_color_trans=trans.trans_input_color,
                           input_mask_trans=trans.trans_input_mask,
                           output_color_trans=trans.trans_output_color)
    

@utils.variable
def ColorNet_val():
    return ColorNetDataset(csv_file='./Data/STEFANN/colornet_val.csv',
                           input_color_trans=trans.trans_input_color,
                           input_mask_trans=trans.trans_input_mask,
                           output_color_trans=trans.trans_output_color)
'''


if __name__ == '__main__':
    # test FANN dataloader
    FANN_data = LightningFANNetDataset()
    FANN_data.setup()
    
    for i in next(iter(FANN_data.train_dataloader())):
        src_img, trgt_label, trgt_img = i
        if len(src_img.shape) >= 2:
            plt.savefig('test.jpg')
    
    
    '''
    for i in ColorNet_train[0]:
        print(i)
        print(i.size())
    '''





# %%
