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



trans_src_img=transforms.Compose([transforms.ToTensor()])
trans_trgt_img=transforms.Compose([transforms.ToTensor()])
trans_trgt_label=transforms.Compose([FANNetLabel()])
trans_input_color=transforms.Compose([transforms.ToTensor()])
trans_input_mask=transforms.Compose([transforms.ToTensor()])
trans_output_color=transforms.Compose([transforms.ToTensor()])

