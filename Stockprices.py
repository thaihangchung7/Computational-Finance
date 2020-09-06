#!/usr/bin/env python
# coding: utf-8

# https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/

# In[1]:


import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import style
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
from plotly import graph_objs as go
import requests


# In[2]:


style.use('ggplot')


# In[3]:


ticker0 = 'SPCE'
ticker1 = 'MRNA'
ticker2 = 'GILD'
ticker3 = 'TSLA'
ticker4 = 'AMD'
ticker5 = 'INTC'

nasdaq = 'NDAQ'
spy = 'SPY'

start = dt.datetime(2015,1,1)
end = dt.datetime.now()


# In[4]:


df0 = web.DataReader(ticker0,"yahoo",start,end)
df1 = web.DataReader(ticker1,"yahoo",start,end)
df2 = web.DataReader(ticker2,"yahoo",start,end)
df3 = web.DataReader(ticker3,"yahoo",start,end)
df4 = web.DataReader(ticker4,"yahoo",start,end)
df5 = web.DataReader(ticker5,"yahoo",start,end)

dfnasdaq = web.DataReader(nasdaq,"yahoo",start,end)
dfspy = web.DataReader(spy,"yahoo",start,end)


# In[5]:


df0.head()
df1.head()
df2.head()
df3.head()
df4.head()
df5.head()


# In[6]:


df0.reset_index(inplace = True)
df1.reset_index(inplace = True)
df2.reset_index(inplace = True)
df3.reset_index(inplace = True)
df4.reset_index(inplace = True)
df5.reset_index(inplace = True)

df0.set_index("Date", inplace = True)
df1.set_index("Date", inplace = True)
df2.set_index("Date", inplace = True)
df3.set_index("Date", inplace = True)
df4.set_index("Date", inplace = True)
df5.set_index("Date", inplace = True)


# In[7]:


#print(df0.head())
#print(df1.head())
#print(df2.head())
#print(df3.head())
#print(df4.head())
#print(df5.head())


# In[8]:


df0['Adj Close'].plot(figsize=(20,20))
plt.title(ticker0)
plt.ylabel('Price')


# In[9]:


def setplt(x = 13, y = 9, a = 1, b = 1):
    f, ax = plt.subplots(a,b,figsize = (x,y))
    sns.despine(f, left = True, bottom = True)
    return f, ax


# In[10]:


f , ax = setplt(30,20,2,3)

#plotting
v_plot = df0['Adj Close'].plot( title = ticker0, ax = ax[0][0])
v_plot = df1['Adj Close'].plot( title = ticker1 ,ax = ax[0][1])
v_plot = df2['Adj Close'].plot( title = ticker2 ,ax = ax[0][2])
v_plot = df3['Adj Close'].plot( title = ticker3 ,ax = ax[1][0])
v_plot = df4['Adj Close'].plot( title = ticker4 ,ax = ax[1][1])
v_plot = df5['Adj Close'].plot( title = ticker5 ,ax = ax[1][2])


# In[11]:


plt.rcParams['figure.figsize'] = (20,10)
ax3 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax4 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax3)

df0['100ma'] = df0['Adj Close'].rolling(window = 10).mean()

ax3.plot(df0['Adj Close'])
ax3.plot(df0['100ma'])
ax4.bar(df0.index,df0['Volume'])


# In[12]:


def hundredavevol(df,ttl,ave):
    plt.rcParams['figure.figsize'] = (20,10)
    ax3 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax4 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax3)

    df['100ma'] = df['Adj Close'].rolling(window = ave).mean()
    
    ax3.title.set_text(ttl)
    ax3.plot(df['Adj Close'])
    ax3.plot(df['100ma'])
    ax4.bar(df.index,df['Volume'])


# In[13]:


hundredavevol(df0,ticker0,20)


# In[14]:


hundredavevol(df1,ticker1,20)


# In[15]:


hundredavevol(df2,ticker2,20)


# In[16]:


hundredavevol(df3,ticker3,20)


# In[17]:


hundredavevol(df4,ticker4,20)


# In[18]:


hundredavevol(df5,ticker5,20)


# In[19]:


hundredavevol(dfnasdaq,nasdaq,20)


# In[20]:


hundredavevol(dfspy,spy,20)


# In[21]:


df0.head()


# In[22]:


df0_ohlc = df0.drop(columns = ['Adj Close','100ma'])
df1_ohlc = df1.drop(columns = ['Adj Close','100ma'])
df2_ohlc = df2.drop(columns = ['Adj Close','100ma'])
df3_ohlc = df3.drop(columns = ['Adj Close','100ma'])
df4_ohlc = df4.drop(columns = ['Adj Close','100ma'])
df5_ohlc = df5.drop(columns = ['Adj Close','100ma'])


# In[23]:


df0_ohlc.columns


# In[24]:


df0_ohlc.reset_index(inplace = True)
df1_ohlc.reset_index(inplace = True)
df2_ohlc.reset_index(inplace = True)
df3_ohlc.reset_index(inplace = True)
df4_ohlc.reset_index(inplace = True)
df5_ohlc.reset_index(inplace = True)

df0_ohlc.set_index("Date", inplace = False)
df1_ohlc.set_index("Date", inplace = False)
df2_ohlc.set_index("Date", inplace = False)
df3_ohlc.set_index("Date", inplace = False)
df4_ohlc.set_index("Date", inplace = False)
df5_ohlc.set_index("Date", inplace = False)


# In[25]:


def candlesticks(ohlc, tick):
    cs = go.Figure(data = [go.Candlestick( 
                    x = ohlc['Date'],
                    open = ohlc['Open'], 
                    high = ohlc['High'],
                    low = ohlc['Low'], 
                    close = ohlc['Close'])])
    cs.update_layout(title = tick)


    cs.show()


# In[26]:


candlesticks(df2_ohlc, ticker2)


# In[27]:


dfnasdaq.head()


# In[28]:


"""
plt.plot(df0['Adj Close']+92)
plt.plot(df5['Adj Close']+47)
plt.plot(dfnasdaq['Adj Close'])
"""


# **The Machine Learning Part**

# from sklearn.model_selection import train_test_split #module for splitting training data and test data
# from sklearn.linear_model import LinearRegression

# train, test = train_test_split(dfnasdaq, test_size = 0.20)

# #plt.plot(df0['Adj Close'])
# 
# print(len(train))
# print(len(dfnasdaq['Close']))

# print(len(dfnasdaq['Close']))
# print(len(dfnasdaq.index))

# #Converting Date to Number of days
# n_days = [i for i in range(len(dfnasdaq.index))]
# #print(n_days)

# #X_train = np.array(train.index).reshape(-1,1)
# X_train = np.array(n_days).reshape(-1,1)
# y_train = train['Close']

# print(len(X_train))
# print(len(y_train))
# 
# dfnasdaq.head()

# #create LinearRegression Object
# model = LinearRegression()
# #Fitting the model
# model.fit(X_train,y_train)

# #Coefficient
# print('Slope = ', np.asscalar(np.squeeze(model.coef_)))
# #Intercept
# print('Intercept = ', model.intercept_)

# plt.figure(1, figsize=(16,10))
# plt.title('Linear Regression | Price vs Time')
# plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
# plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
# plt.xlabel('Integer Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# In[38]:


plt.plot(df2['Adj Close'])


# In[ ]:





# In[ ]:




