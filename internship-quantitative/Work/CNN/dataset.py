from torch.utils.data import Dataset
import os 
import pandas as pd
import re
from PIL import Image
from torchvision.io import read_image

class dataset(Dataset):
    '''
    自定义的dataset类,其继承的父类来自torch.utils.data。它是实现torch.utils.data.DataLoader的必须参数。
    该类将根据提供的begintime和endtime限定获取数据时数据的对应时间。
    begintime:数据集图片对应的起始时间
    endtime:数据集图片的终止时间
    path:图片数据存放的路径
    target_transform:将label从字符串转换为torch张量的函数
    '''
    def __init__(self,begintime,endtime,path,target_transform,transform=None):
        super(Dataset).__init__()
        self.path=path
        allimagelist=os.listdir(path)
        begintime=pd.Timestamp(begintime)
        endtime=pd.Timestamp(endtime)
        timagelist=[]
        labellist=[]
        for i in allimagelist:
            label=int(re.search(pattern="^\d",string=i).group())
            try:
                date=re.search("\d{4}-\d{2}-\d{2}",string=i).group()
            except:
                pass
            else:
                try:
                    date=pd.Timestamp(date)
                    if date>=begintime and date<=endtime:
                        timagelist.append(i)
                        labellist.append(label)
                except:
                    print(date)
        self.imagelist=timagelist
        self.labellist=labellist
        self.transform=transform
        self.target_transform=target_transform
    
    def __getitem__(self, index):
        img_name=self.imagelist[index]
        img=Image.open(self.path+img_name)
        img=img.convert('RGB')
        if self.transform:
            img=self.transform(img)
        else:
            NotImplementedError("Must be preprocessed through transforms")
        label=self.labellist[index]
        if self.target_transform:
            label=self.target_transform(label)
        return (img,label,img_name)
    
    def __len__(self):
        return len(self.labellist)
        