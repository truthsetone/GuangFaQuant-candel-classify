'''
网络训练全过程
'''
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Resize,Normalize,Compose
from torchvision.io import read_image
from importlib import reload
from torch.optim.adamw import AdamW
import torch
import matplotlib.pyplot as plt
import logging
reload(logging)
import re
import pandas as pd
import numpy as np
from model import model
from early_stop import earlystop
from dataset import dataset
def labeltrans(y):
    '''
    用于将int标签转换为指定位置张量的函数，是Dataset需要的参数
    '''
    x=torch.zeros(3,dtype=torch.float)
    y=torch.tensor(y)
    value=torch.tensor(1.0)
    x=x.scatter_(index=y,src=value,dim=0)
    return x

def evaluate(df_predict,path_output,num_group=10):
    '''
    用于评价获得的因子的函数。最终的评价指标将位于：path_output中。
    df_predict：得到的因子
    path_output:输出文件
    num_group:用于指定分组的个数。
    '''
    dict_output = {} # dict, store all evaluation results
    folder_data = '/data/AI_Project'
    
    df_label = pd.read_hdf(f"/data/AI_Project/Labels/Ret20.h5")
    df_label['labelDate']=pd.to_datetime(df_label["labelDate"],format="%Y%m%d")
    ds_label_1 = df_label.set_index(['labelDate', 'code'])['labelValue']
    ds_label_1.index.names = ['date', 'code']
    ds_label_1.name = 'label'
    
    df_label_2 = pd.read_hdf(f"/data/AI_Project/Labels/Ret5.h5")
    df_label_2['labelDate']=pd.to_datetime(df_label["labelDate"],format="%Y%m%d")
    ds_label_2 = df_label_2.set_index(['labelDate', 'code'])['labelValue']
    ds_label_2.index.names = ['date', 'code']
    ds_label_2.name = 'label'
    
    index_intersection = df_predict.index.intersection(ds_label_1.index)
    df_predict = df_predict.reindex(index_intersection)
    df_predict['mean'] = df_predict.mean(axis=1)
    df_predict=pd.DataFrame(df_predict,dtype=float)

    ds_label = ds_label_1.reindex(index_intersection)
    ds_label=pd.Series(ds_label,dtype=float)
    
    #ic,icir
    list_ds_ic = []
    nd_date = df_predict.index.levels[0].to_numpy()
    nd_date=np.array(nd_date)
    for _date in nd_date:
        list_ds_ic.append(df_predict.loc[_date].corrwith(ds_label.loc[_date], method='spearman'))
    df_ic = pd.concat(list_ds_ic, axis=1, keys=nd_date, names='date').T
    dict_output['ic'] = df_ic.round(3)
    dict_output['ic_mean'] = df_ic.mean().round(3)
    dict_output['ic_ir'] = (df_ic.mean() / df_ic.std()).round(3)

    # corr
    df_corr = df_predict.corr(method='pearson')
    np.fill_diagonal(df_corr.to_numpy(), 0)

    dict_output['corr'] = df_corr.round(3)
    
    # group alpha
    ds_alpha = ds_label_2 - ds_label_2.groupby('date').mean()
    list_ds_alpha_group = []
    for _factor in df_predict.columns:
        ds_predict = df_predict[_factor]
        ds_predict_divide = ds_predict.groupby('date').apply(lambda x: pd.qcut(x, q=num_group, labels=False, duplicates='drop'))
        ds_alpha_group = ds_alpha.groupby('date').apply(lambda x: x.groupby(ds_predict_divide).mean())
        list_ds_alpha_group.append(ds_alpha_group)
    df_alpha_group = pd.concat(list_ds_alpha_group, axis=1, keys = df_predict.columns)
    df_alpha_group.index.names = ['date', 'group']

    dict_output['group_alpha'] = df_alpha_group.round(4)
    dict_output['group_alpha_mean'] = df_alpha_group.groupby('group').mean().T.round(4)
    dict_output['group_alpha_ir'] = (df_alpha_group.groupby('group').mean() / df_alpha_group.groupby('group').std()).T.round(3)

    # store
    with pd.ExcelWriter(path_output) as writer:
        for _key, _value in dict_output.items():
            _value.to_excel(writer, sheet_name=_key)
            
    output=pd.read_excel(path_output,sheet_name='ic_mean')
    logging.info(f"\n{output.iloc[:,1:]}")  
    
    output=pd.read_excel(path_output,sheet_name='group_alpha_mean')
    logging.info(f"\n{output.iloc[:,2:]}")  

def modelinit(layer):
    '''
    初始化网络函数
    '''
    if type(layer)==torch.nn.Conv2d:
        #torch.nn.init.xavier_normal_(layer.weight)
        torch.nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
        pass
    if type(layer)==torch.nn.Linear:
        pass
        #torch.nn.init.uniform_(layer.weight,a=-0.1,b=0.1)
        #torch.nn.init.constant_(layer.bias,0.1)

train_years=['2006-01-01']
image_transform=Compose(
    [Resize([320,320]),
    ToTensor(),
    Normalize([0.951,0.925,0.91],[0.178,0.214,0.256])]
    )
for year in train_years:
    year=pd.Timestamp(year)
    batchsize=32
    train_dataset=dataset(path="/home/hydu/CNN/data/image/",begintime=year,endtime=year+pd.Timedelta(days=3650),target_transform=labeltrans,transform=image_transform)
    train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=6)
    val_dataset=dataset(path="/home/hydu/CNN/data/image/",begintime=year+pd.Timedelta(days=3680),endtime=year+pd.Timedelta(days=5140),target_transform=labeltrans,transform=image_transform)
    val_dataloader=DataLoader(val_dataset,batch_size=32,shuffle=True,num_workers=6)
    test_dataset=dataset(path="/home/hydu/CNN/data/image/",begintime=year+pd.Timedelta(days=5170),endtime=year+pd.Timedelta(days=6235),target_transform=labeltrans,transform=image_transform)
    test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=6)
    #指定记录文件和格式
    logging.basicConfig(filename='/home/hydu/CNN/log/log_medium.txt',
                        format = '%(asctime)s - %(message)s',
                        level=logging.INFO)
    
    a=model()
    early=earlystop(path="/home/hydu/CNN/",patience=2)
    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        device=torch.device("cpu")
    a.apply(modelinit)
    a.to(device)


    epoch=10
    LR=1e-7
    trainloss_value=0
    optimizer=AdamW(a.parameters(),lr=LR,weight_decay=5e-4)
    lossfunc=torch.nn.CrossEntropyLoss() 

    logging.info(f"每日0.33标签,10,4,2,2006三通道")

#训练与验证过程
    for i in range(epoch):
        a.train()#使用了dropout
        all_loss=0
        trainloss_value=0
        for index,(image,label,name)  in enumerate(train_dataloader):
            image=image.to(device)
            label=label.to(device)
            output=a.forward(image).to(device)
            loss=lossfunc(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for p in optimizer.param_groups:
            p['lr']*=0.95
            
        a.eval()
        count=0
        val_loss=0
        codes=[]
        times=[]
        predict_prob=[]
        for index,(image,label,names)  in enumerate(val_dataloader):
            for name in names:
                real_code=re.search(pattern="\d{6}",string=name).group()
                real_day=re.search(pattern="\d{4}-\d{2}-\d{2}",string=name).group()
                codes.append(real_code)
                times.append(real_day)
            image=image.to(device)
            label=label.to(device)
            output=a.forward(image).to(device)
            loss=lossfunc(output,label)
            output_indice=torch.max(output,dim=1)[1]
            label_indice=torch.max(label,dim=1)[1]
            count+=(output_indice==label_indice).sum().item()
            for each in output:
                predict_prob.append(round(float(each[2]),4))
        logging.info(f"Epoch {i} vanlidation accuracy: {round(float(count/(val_dataloader.__len__()*batchsize)),4)}")
        logging.info(f"Vanlidation loss epoch {i} : {round(val_loss,4)}")
        df_predict=pd.DataFrame([times,codes,predict_prob]).T
        df_predict.columns=["date","code","0"]
        df_predict['date']=pd.to_datetime(df_predict["date"],format="%Y-%m-%d")
        df_predict=df_predict.set_index(["date","code"]).sort_index()
        df_predict_1=pd.DataFrame(df_predict,dtype=float)
        #df_predict_1.to_excel(f"/home/hydu/CNN/result/van_factor_{year}.xlsx")(不能包含过多的行：1048576)
        evaluate(df_predict_1,f"/home/hydu/CNN/evaluation_van_epoch{i}.xlsx")
        if early.check_stop(a,-count)==False:
            break
        
    #测试过程    
    a.load_state_dict(torch.load("/home/hydu/CNN/best_network.pth"))
    a.eval()
    count=0
    test_loss=0
    codes=[]
    times=[]
    predict_prob=[]
    for index, (image,label,names) in enumerate(test_dataloader):
        for name in names:
            real_code=re.search(pattern="\d{6}",string=name).group()
            real_day=re.search(pattern="\d{4}-\d{2}-\d{2}",string=name).group()
            codes.append(real_code)
            times.append(real_day)
        image=image.to(device)
        label=label.to(device)
        output=a.forward(image).to(device)
        loss=lossfunc(output,label)
        output_indice=torch.max(output,dim=1)[1]
        label_indice=torch.max(label,dim=1)[1]
        count+=(output_indice==label_indice).sum().item()
        for each in output:
            predict_prob.append(round(float(each[2]),4))
    logging.info(f"Testset accuracy: {float(count/(test_dataloader.__len__()*batchsize))}")
    logging.info(f"Testset loss : {test_loss}")

    df_predict=pd.DataFrame([times,codes,predict_prob]).T
    df_predict.columns=["date","code","0"]
    df_predict['date']=pd.to_datetime(df_predict["date"],format="%Y-%m-%d")
    df_predict=df_predict.set_index(["date","code"]).sort_index()
    df_predict.to_excel(f"/home/hydu/CNN/result/test_factor_{year}.xlsx")
    df_predict_1=pd.DataFrame(df_predict,dtype=float)
    evaluate(df_predict_1,path_output="/home/hydu/CNN/evaluation_test.xlsx")