'''
用于计算数据集图片各个通道的均值和标准差的代码
'''


from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToTensor,Resize,Normalize,Compose
import logging
from torch.utils.data import Dataset
import os 
import pandas as pd
import re
from PIL import Image

class dataset(Dataset):
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
        

def labeltrans(y):
    x=torch.zeros(3,dtype=torch.float)
    y=torch.tensor(y)
    value=torch.tensor(1.0)
    x=x.scatter_(index=y,src=value,dim=0)
    return x

image_transform=Compose([Resize([320,320]),ToTensor()])

train_dataset=dataset(path="/home/hydu/CNN/data/image/",begintime="2007-01-01",endtime="2019-12-31",target_transform=labeltrans,transform=image_transform)
train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)

logging.basicConfig(filename='/home/hydu/CNN/log/log_medium.txt',
                    format = '%(asctime)s - %(message)s',
                    level=logging.INFO)


batchsize=32
N=0
a=0
b=0
c=0
a2=0
b2=0
c2=0
for index,(image,name,label) in enumerate(train_dataloader):
    N+=len(name)
    tmp_matrix=image[:,0,:,:]
    a+=torch.sum(tmp_matrix)
    a2+=torch.sum(torch.float_power(tmp_matrix,2))
    
    tmp_matrix=image[:,1,:,:]
    b+=torch.sum(tmp_matrix)
    b2+=torch.sum(torch.float_power(tmp_matrix,2))
        
    tmp_matrix=image[:,2,:,:]
    c+=torch.sum(tmp_matrix)
    c2+=torch.sum(torch.float_power(tmp_matrix,2))
        
  
   
N=320*320*N
a=float(a/(N))
b=float(b/(N))
c=float(c/(N))

logging.info(f"{a},{b},{c}")

a2=torch.sqrt(a2/N-a*a)
b2=torch.sqrt(b2/N-b*b)
c2=torch.sqrt(c2/N-c*c)

logging.info(f"{a2},{b2},{c2}")
