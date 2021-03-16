import config
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import transformation
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




@utils.variable
def FANNet_train():
    return FANNetDataset(csv_file='./Data/STEFANN/fannet_train.csv',
                         src_img_trans=transformation.trans_fannet_src_img,
                         trgt_label_trans=transformation.trans_fannet_trgt_label,
                         trgt_img_trans=transformation.trans_fannet_trgt_img)



if __name__=='__main__':
    print(FANNet_train[0])




