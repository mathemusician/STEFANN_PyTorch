import torch
import argparse
from PIL import Image
import os.path as path
import torchvision.transforms as transforms
import model
import utils
import transformation



MODEL='FANNet'
MODEL_DIR='./Model/FANNet'
SRC_IMG='/mnt/Data/GoogleFontsSTEFANN/fannet/train/ibmplexmono-thinitalic/78.jpg'
TRGT_LABEL=ord('J')-ord('A')
OUTPUT_IMG='./output.jpg'
EPOCH=2

if __name__=='__main__':
    if MODEL=='FANNet':
        net=torch.nn.DataParallel(model.FANNet()).cuda()
        utils.load_model(net,
                         path.join(MODEL_DIR,str(EPOCH)),
                         MODEL+'.pth')
        src_img=Image.open(SRC_IMG).convert('L')
        trgt_label=TRGT_LABEL
        src_img=transformation.trans_src_img(src_img)
        trgt_label=transformation.trans_trgt_label(trgt_label)
        src_img=torch.unsqueeze(src_img,0)
        trgt_label=torch.unsqueeze(trgt_label,0)
        output_img=net(src_img,trgt_label)
        output_img.cpu()
        output_img=torch.squeeze(output_img,0)
        trans_ToPILImage=transforms.ToPILImage()
        output_img=trans_ToPILImage(output_img)
        output_img.save(OUTPUT_IMG)
