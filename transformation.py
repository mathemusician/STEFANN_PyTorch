import torch
import torchvision.transforms as transforms
import config



LEN_TRGT_CHRS=len(config.TRGT_CHRS)



class FANNetLabel(object):
    def __init__(self):
        super().__init__()

    
    def __call__(self,label):
        onehot=torch.LongTensor([0]*LEN_TRGT_CHRS)
        onehot[label]=1
        return onehot



trans_fannet_src_img=transforms.Compose([transforms.ToTensor()])
trans_fannet_trgt_img=transforms.Compose([transforms.ToTensor()])
trans_fannet_trgt_label=transforms.Compose([FANNetLabel()])
transformation_colornet_train=transforms.Compose([transforms.ToTensor()])
transformation_colornet_val=None

