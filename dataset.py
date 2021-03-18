import config
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import transformation as trans
import config
import utils



class FANNetDataset(data.Dataset):
    def __init__(self,csv_file,
                 src_img_trans=None,
                 trgt_label_trans=None,
                 trgt_img_trans=None):
        super().__init__()
        self.df=pd.read_csv(csv_file)
        self.src_img_trans=src_img_trans
        self.trgt_label_trans=trgt_label_trans
        self.trgt_img_trans=trgt_img_trans


    def __len__(self):
        return len(self.df)

    
    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()
            
        info=self.df.loc[index,:]
        src_img_path=info[0]
        trgt_label=int(info[1])
        trgt_img_path=info[2]
        src_img=Image.open(src_img_path).convert('L')
        trgt_img=Image.open(trgt_img_path).convert('L')

        if self.src_img_trans!=None:
            src_img=self.src_img_trans(src_img)

        if self.trgt_label_trans!=None:
            trgt_label=self.trgt_label_trans(trgt_label)

        if self.trgt_img_trans!=None:
            trgt_img=self.trgt_img_trans(trgt_img)

        return src_img,trgt_label,trgt_img



class ColorNetDataset(data.Dataset):
    def __init__(self,csv_file,
                 input_color_trans=None,
                 input_mask_trans=None,
                 output_color_trans=None):
        super().__init__()
        self.df=pd.read_csv(csv_file)
        self.input_color_trans=input_color_trans
        self.input_mask_trans=input_mask_trans
        self.output_color_trans=output_color_trans


    def __len__(self):
        return len(self.df)


    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        info=self.df.loc[index,:]
        input_color_img_path=info[0]
        input_mask_img_path=info[1]
        output_color_img_path=info[2]
        input_color_img=Image.open(input_color_img_path).convert('RGB')
        input_mask_img=Image.open(input_mask_img_path).convert('L')
        output_color_img=Image.open(output_color_img_path).convert('RGB')

        if self.input_color_trans!=None:
            input_color_img=self.input_color_trans(input_color_img)

        if self.input_mask_trans!=None:
            input_mask_img=self.input_mask_trans(input_mask_img)

        if self.output_color_trans!=None:
            output_color_img=self.output_color_trans(output_color_img)

        return input_color_img,input_mask_img,output_color_img



@utils.variable
def FANNet_train():
    return FANNetDataset(csv_file='./Data/STEFANN/fannet_train.csv',
                         src_img_trans=trans.trans_src_img,
                         trgt_label_trans=trans.trans_trgt_label,
                         trgt_img_trans=trans.trans_trgt_img)


@utils.variable
def FANNet_val():
    return FANNetDataset(csv_file='./Data/STEFANN/fannet_val.csv',
                         src_img_trans=trans.trans_src_img,
                         trgt_label_trans=trans.trans_trgt_label,
                         trgt_img_trans=trans.trans_trgt_img)


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



if __name__=='__main__':
    for i in FANNet_train[0]:
        print(i)
        print(i.size())

    for i in ColorNet_train[0]:
        print(i)
        print(i.size())

