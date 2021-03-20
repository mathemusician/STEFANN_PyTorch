import os
import os.path as path
import itertools
import pandas as pd
import glob
import config
import random



FANNET_IMG_DIR='/mnt/Data/GoogleFontsSTEFANN/fannet'
COLORNET_IMG_DIR='/mnt/Data/GoogleFontsSTEFANN/colornet'


def process_fannet(img_dir,save_path):
    perms=itertools.product(os.listdir(img_dir),
                            list(config.SRC_CHRS),
                            list(config.TRGT_CHRS))
    df=pd.DataFrame(columns=['src_img_path',
                             'trgt_label',
                             'trgt_img_path',
                             'font'])
    item_list=[]

    for perm in perms:
        item={}
        src_chr=str(ord(perm[1]))
        trgt_chr=str(ord(perm[2]))
        font=perm[0]
        src_img_path=path.join(img_dir,font,src_chr+'.jpg')
        trgt_img_path=path.join(img_dir,font,trgt_chr+'.jpg')
        item['src_img_path']=src_img_path
        item['trgt_label']=config.TRGT_CHRS.find(perm[2])
        item['trgt_img_path']=trgt_img_path
        item['font']=font
        item_list.append(item)

    random.shuffle(item_list)

    df=df.append(item_list,ignore_index=True)
    df.to_csv(save_path,index=False)


def process_colornet(img_dir,save_path):
    input1_img_dir=path.join(img_dir,'input_color')
    input2_img_dir=path.join(img_dir,'input_mask')
    output_img_dir=path.join(img_dir,'output_color')
    files_name=sorted(p.split('/')[-1] for p in \
                      glob.glob('{}/*{}'.format(input1_img_dir,'.jpg')))
    df=pd.DataFrame(columns=['input_color',
                             'input_mask',
                             'output_color'])
    item_list=[]

    for file_name in files_name:
        item={}
        item['input_color']=path.join(input1_img_dir,file_name)
        item['input_mask']=path.join(input2_img_dir,file_name)
        item['output_color']=path.join(output_img_dir,file_name)
        item_list.append(item)

    random.shuffle(item_list)

    df=df.append(item_list,ignore_index=True)
    df.to_csv(save_path,index=False)



if __name__=='__main__':
    process_fannet(path.join(FANNET_IMG_DIR,'train'),
                   './Data/STEFANN/fannet_train.csv')
    process_fannet(path.join(FANNET_IMG_DIR,'valid'),
                   './Data/STEFANN/fannet_val.csv')
    process_colornet(path.join(COLORNET_IMG_DIR,'train'),
                     './Data/STEFANN/colornet_train.csv')
    process_colornet(path.join(COLORNET_IMG_DIR,'valid'),
                     './Data/STEFANN/colornet_val.csv')

