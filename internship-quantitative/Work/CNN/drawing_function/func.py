import mplfinance as mlf 
import pandas as pd
import matplotlib.pyplot as plt
import drawing_function
from joblib import Parallel,delayed
import copy

def get_macd_colors(data):
    colors=[]
    for i in range(len(data)):
        if data[i]<0:
            colors.append("green")  
        elif data[i]>0:
            colors.append("red") 
        else:
            colors.append("#F0FFFF")
    return colors 

def subdrawing(data,s,profit,beginyear,endyear):
    labelDate=data.index[20]
    drawing_begin=data.index[0]
    beginday=pd.Timestamp(str(beginyear-1)+"-12-01")
    if beginday<=drawing_begin:
        code=data.iloc[0,0]
        labelv=profit.query('(labelDate == @labelDate)')
        data=data.iloc[1:21,:]
        drawing_list=[
            mlf.make_addplot(data['MA5'],type='line',color='b',panel=0),
            mlf.make_addplot(data['MA10'],type='line',color='r',panel=0),
            mlf.make_addplot(data['MA15'],type='line',color='g',panel=0),
            mlf.make_addplot(data['MA20'],type='line',color='y',panel=0),
            mlf.make_addplot(data['MACD'],type='line',panel=2,color='b'),
            mlf.make_addplot(data['Signal'],type='line',panel=2,color='y'),
            mlf.make_addplot(data['Histogram'],type='bar',panel=2,color=get_macd_colors(data['Histogram']))
        ]
        fig,axes=mlf.plot(
            data,type='candle',
            volume=True,ylabel="",
            style=s,addplot=drawing_list,
            figscale=3,returnfig=True,panel_ratios=(5,3,2)
        )
        axes[0].yaxis.set_ticks([])
        axes[2].yaxis.set_ticks([])
        axes[2].set_ylabel("")
        axes[4].yaxis.set_ticks([])
        axes[5].xaxis.set_ticks([])
        axes[5].yaxis.set_ticks([])
        for i in range(6):
            axes[i].spines[['top','bottom','left','right']].set_visible(False)
        try:
            labelv=float(labelv.iloc[0,1])
        except:
            #print(labelDate,code)
            plt.close() 
        else:
            path="/home/hydu/CNN/data/image/"
            if labelv>0:
                name=f"2-{code}-{labelDate}.png"
                path=path+name
            elif labelv==0:
                name=f"1-{code}-{labelDate}.png"
                path=path+name
            else:
                name=f"0-{code}-{labelDate}.png"
                path=path+name
            plt.savefig(fname=path,dpi=20)  
            plt.close()  

def drawing(this_data,profit20,code,style,beginyear,endyear): #注意指标的计算，数据必须从2006年开始，即使仅仅画其他年的图
    #this_data=copy.copy(data.query('code == @code'))
    MA5=this_data['Close'].rolling(window=5).mean()
    MA10=this_data['Close'].rolling(window=10).mean()
    MA15=this_data['Close'].rolling(window=15).mean()
    MA20=this_data['Close'].rolling(window=20).mean()
    Ema12=this_data['Close'].ewm(span=12,adjust=False).mean()
    Ema26=this_data['Close'].ewm(span=26,adjust=False).mean()
    MACD=Ema12-Ema26
    Signal=MACD.ewm(span=9,adjust=False).mean()
    Histogram=MACD-Signal
    Histogram.columns=['Macd']
    this_data['Signal']=Signal
    this_data['MACD']=MACD
    this_data['Histogram']=Histogram
    this_data['MA5']=MA5
    this_data['MA10']=MA10
    this_data['MA15']=MA15
    this_data['MA20']=MA20
    row,line=this_data.shape
    for i in range(0,row-22):
        if i<=20:  #如果时间不到20天，平均线不存在
            continue
        try:
            slice_data=this_data.iloc[i:i+22,:]
            #print(slice_data)
            subdrawing(data=slice_data,s=style,profit=profit20,beginyear=beginyear,endyear=endyear)
        finally:
            pass
        
    