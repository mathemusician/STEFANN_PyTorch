import torch
import torch.nn as nn
import config



class FANNet(nn.Module):
    def __init__(self,onehot_len=len(config.TRGT_CHRS),input_shape=(64,64)):
        super().__init__()
        self.input_shape=input_shape
        self.onehot_len=onehot_len
        self.conv=nn.Sequential(self._conv3x3(1,16),
                                self._conv3x3(16,16),
                                self._conv3x3(16,1))
        self.flatten=nn.Flatten()
        self.img_linear=nn.Linear(input_shape[0]*input_shape[1],512)
        self.onehot_linear=nn.Linear(self.onehot_len,512)
        self.nolinear=nn.Sequential(nn.Linear(1024,1024),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5),
                                    nn.Linear(1024,1024),
                                    nn.ReLU(inplace=True))
        self.upsample=nn.Sequential(nn.Unflatten(dim=1,unflattened_size=(16,8,8)),
                                    self._up_block(16,16),
                                    self._up_block(16,16),
                                    self._up_block(16,1))


    def _conv3x3(self,inplace_channel,output_channel):
        layers=[nn.Conv2d(inplace_channel,output_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='replicate'),
                nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


    def _up_block(self,input_channel,output_channel):
        layers=[nn.Upsample(scale_factor=2),
                nn.Conv2d(input_channel,output_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='replicate'),
                nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


    def forward(self,src_img,trgt_one_hot):
        img_feat=self.conv(src_img)
        img_feat=self.img_linear(self.flatten(img_feat))
        trgt_feat=self.onehot_linear(trgt_one_hot)
        feat=torch.cat((img_feat,trgt_feat),dim=1)
        feat=self.nolinear(feat)
        trgt_img=self.upsample(feat)
        return trgt_img



class ColorNet(nn.Module):
    def __init__(self,input_shape=(64,64)):
        super().__init__()
        self.input_shape=input_shape
        self.color_conv=self._conv3x3(3,64,norm='batch')
        self.mask_conv=self._conv3x3(1,64,norm='batch')
        self.conv=nn.Sequential(self._conv3x3(128,64),
                                self._conv3x3(64,64),
                                nn.MaxPool2d(kernel_size=2),
                                self._conv3x3(64,128),
                                self._conv3x3(128,128),
                                nn.MaxPool2d(kernel_size=2),
                                self._conv3x3(128,256),
                                self._conv3x3(256,256),
                                self._conv3x3(256,256))
        self.upsample=nn.Sequential(self._up_block(256,128),
                                    self._up_block(128,64),
                                    self._conv3x3(64,3))
        

    def _conv3x3(self,input_channel,output_channel,norm='none'):
        assert norm in ['none','batch'],\
               'the value of norm should be in [none,batch]'

        layers=[nn.Conv2d(input_channel,output_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='replicate'),
                nn.LeakyReLU(negative_slope=0.2,inplace=True)]

        if norm=='batch':
            layers.append(nn.BatchNorm2d(output_channel))
        
        return nn.Sequential(*layers)


    def _up_block(self,input_channel,output_channel):
        layers=[nn.Upsample(scale_factor=2),
                nn.Conv2d(input_channel,output_channel,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='replicate'),
                nn.LeakyReLU(negative_slope=0.2,inplace=True)]
        return nn.Sequential(*layers)


    def forward(self,input_color,input_mask):
        color_feat=self.color_conv(input_color)
        mask_feat=self.mask_conv(input_mask)
        feat=torch.cat((color_feat,mask_feat),dim=1)
        feat=self.conv(feat)
        output_color=self.upsample(feat)
        return output_color



if __name__=='__main__':
    fannet=FANNet()
    colornet=ColorNet()

