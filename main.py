import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import talib
from talib import RSI, BBANDS
from sklearn.linear_model import LinearRegression


# Figure bitcoin
bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates= True)
bitcoin = bitcoin.dropna()
bitcoin.head()
bitcoin['Close'].plot()
plt.show()


# Essai ta-lib avec moyenne glissante
Bc = talib.SMA(bitcoin['Close'].values, timeperiod=30)
plt.plot(Bc, label='SMA Bitcoin')
plt.plot(bitcoin['Close'].values,label='Price')
plt.legend(loc='best')
plt.show()


# Essai ta-lib 2 avec RSI et BBands
up, mid, low = BBANDS(bitcoin['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
rsi = RSI(bitcoin['Close'].values, timeperiod=14)
print("RSI (first 10 elements)\n", rsi[14:24])


def bbp_fonc(donnees):
    bbp=[]
    for i in range (len(donnees['Close'])):
        bbp += [(donnees['Close'][i] - low[i]) / (up[i] - low[i])]
    return bbp

print(bbp_fonc(bitcoin)[20:40])

# Création d'un dataframe du RSI pour diff valeurs de 'timeperiod'
def rsi_frame(donnees, n):
    Rsi_f = []
    c = ['period ='+str( i) for i in range (2,n+1)]
    for i in range(2,n+1):
        Rsi_f.append(RSI(donnees['Close'].values, timeperiod=i))
    Rsi_ft = np.transpose(Rsi_f)
    Rsi_frame = pd.DataFrame(Rsi_ft, index= donnees.index.values ,columns=c )
    return Rsi_frame


Rindice = rsi_frame(bitcoin, 20)
print(Rindice.head())
print(Rindice)


##mettre bitcoin et moyenne glissante sur le mm graphique ac la date en abs
##faire les frames des autres indicateurs du mail
