import torch
import torch.nn as nn



class MAELoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.mae=nn.MAELoss()


    def forward(self,output_img,trgt_img):
        return self.mae(output_img,trgt_img)

