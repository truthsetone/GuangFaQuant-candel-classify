import mplfinance as mlf 
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel,delayed
import copy
import sys

sys.path.append("/home/hydu/CNN")

from drawing_function import func

#因为计算指标的问题，数据必须从2006年开始获得
path="/data/AI_Project/DataSets/AdjDaily/"
beginfile="/data/AI_Project/DataSets/AdjDaily/2006.h5"
beginyear=2021
endyear=2022
filenames=[]
for i in range(2007,endyear+1):
    filename=f"{path}{i}.h5"
    filenames.append(filename)
data=pd.read_hdf(beginfile,key='data')
for filename in filenames:
    tempdata=pd.read_hdf(filename,key='data')
    data=pd.concat([data,tempdata])
profit20=pd.read_hdf("/data/AI_Project/Labels/Ret20.h5")
stock_code=sorted(list(data.loc[:,'code'].unique()),key=str.lower)

data.index=pd.to_datetime(data.index,format="%Y%m%d")
profit20.index=pd.to_datetime(profit20.index,format="%Y%m%d")
profit20['labelDate']=pd.to_datetime(profit20["labelDate"],format="%Y%m%d")
data.columns=columns=['code', 'preclose', 'Open', 'High', 'Low', 'Close', 'Volume', 'amount','trdnum']

mc=mlf.make_marketcolors(up='r',down='g',wick={'up':'blue','down':'orange'},volume={'up':'r','down':'g'})
s=mlf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mc,mavcolors=['r','g','b','orange'],gridstyle='')

data_list=[]
profit_list=[]
for code in stock_code:
    temp_data=copy.copy(data.query('code == @code'))
    temp_profit=copy.copy(profit20.query('code == @code'))
    data_list.append(temp_data)
    profit_list.append(temp_profit)

Parallel(n_jobs=75)(delayed(func.drawing)(data,profit,style=s,code=code,beginyear=beginyear,endyear=endyear) for data,profit,code in zip(data_list,profit_list,stock_code))
