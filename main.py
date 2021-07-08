import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import talib
from talib import RSI, BBANDS, ROC, SMA, EMA, ATR, ADX, CCI, ROC, WILLR, STOCH
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import time


# Figure bitcoin
bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates= True)
bitcoin = bitcoin.dropna()
bitcoin.head()
bitcoin['Close'].plot()
plt.show()


# Essai ta-lib avec moyenne glissante
Bc = talib.SMA(bitcoin['Close'], timeperiod=30)
plt.plot(bitcoin['Close'],label='Price')
plt.plot(Bc, label='SMA Bitcoin')
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
#print(Rindice.head())
#print("RSI dataframe\n",Rindice)


# Création d'un dataframe du ROC pour diff valeurs de 'timeperiod'
def roc_frame(donnees, n):
    Roc_f = []
    c = ['period ='+str( i) for i in range (2,n+1)]
    for i in range(2,n+1):
        Roc_f.append(ROC(donnees['Close'].values, timeperiod=i))
    Roc_ft = np.transpose(Roc_f)
    Roc_frame = pd.DataFrame(Roc_ft, index= donnees.index.values ,columns=c )
    return Roc_frame

Roc_indice = roc_frame(bitcoin, 20)
#print(Roc_indice.head())
#print("ROC dataframe\n",Roc_indice)


## Essai algo de classification
Btc = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates= True)
Btc = Btc.dropna()
Btc = Btc.iloc[::3,:]  #permet de sauter des jours

Btc['sma_10'] = SMA(Btc['Close'], timeperiod=10)
Btc['sma_20'] = SMA(Btc['Close'], timeperiod=20)
Btc['sma_50'] = SMA(Btc['Close'], timeperiod=50)
Btc['sma_100'] = SMA(Btc['Close'], timeperiod=100)
Btc['sma_200'] = SMA(Btc['Close'], timeperiod=200)

Btc['ema_10'] = EMA(Btc['Close'], timeperiod=10)
Btc['ema_20'] = EMA(Btc['Close'], timeperiod=20)
Btc['ema_50'] = EMA(Btc['Close'], timeperiod=50)
Btc['ema_100'] = EMA(Btc['Close'], timeperiod=100)
Btc['ema_200'] = EMA(Btc['Close'], timeperiod=200)

Btc['ATR'] = ATR(Btc['High'].values, Btc['Low'].values, Btc['Close'].values, timeperiod=14) #pouvoir changer timep
Btc['ADX'] = ADX(Btc['High'].values, Btc['Low'].values, Btc['Close'].values, timeperiod=14)
Btc['CCI'] = CCI(Btc['High'].values, Btc['Low'].values, Btc['Close'].values, timeperiod=14)
Btc['ROC'] = ROC( Btc['Close'].values, timeperiod=14)
Btc['RSI'] = RSI( Btc['Close'].values, timeperiod=14)
Btc['WILLR'] = WILLR(Btc['High'].values, Btc['Low'].values, Btc['Close'].values, timeperiod=14)
#Btc['STOCH'] = STOCH(Btc['High'].values, Btc['Low'].values, Btc['Close'].values)
Btc = Btc.dropna()
print(Btc.head(-1))

# construction de signaux à utiliser à la place des indicateurs
# par ex: Rsi>70 -> 0 et Rsi<30 -> 1 ou SMA_10>SMA_50 -> 1 et SMA_10<SMA_50 -> 0

Btc['pred_price']= np.where(Btc['Close'].shift(-1)> Btc['Close'], 1, 0)


# on sépare les données en train (x) et test (y)
x = Btc.drop(columns='pred_price')
y = Btc['pred_price']

train_x = x['2015-04-08':'2019-09-21']  #début, milieu et fin
test_x = x['2019-09-22':'2021-06-21']
train_y = y['2015-04-08':'2019-09-21']
test_y = y['2019-09-22':'2021-06-21']


# on normalise les données
scaler = MinMaxScaler(feature_range=(0,1))
train_x_scaled = scaler.fit_transform(train_x)
print("les éléments normalisés\n",train_x_scaled)
print("\n")


# on utilise les modèles de classification
dict_classifiers = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=100),
    "Nearest Neighbors": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(solver='adam', alpha=0.0001,
                                learning_rate='constant', learning_rate_init=0.001),
    "Naive Bayes": GaussianNB(),
}
#"Support Vector Machine": SVC(gamma='auto'),
nb_classifiers = len(dict_classifiers.keys())

def batch_classify(train_x_scaled, train_y, verbose= True):
    Btc_results= pd.DataFrame(data= np.zeros(shape=(nb_classifiers, 3)),
                              columns= ['classifier', 'train_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.process_time()
        classifier.fit(train_x_scaled, train_y)
        t_end = time.process_time()
        t_diff = t_end - t_start
        train_score = classifier.score(train_x_scaled, train_y)
        Btc_results.loc[count, 'classifier'] = key
        Btc_results.loc[count, 'train_score'] =train_score
        Btc_results.loc[count, 'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count+=1
    return Btc_results

Btc_results = batch_classify(train_x_scaled, train_y)
print("\n Results classifiers\n", Btc_results.sort_values(by='train_score'))


# on exécute notre stratégie sur la partie test
test_x_scaled = scaler.fit_transform(test_x)
print("\ntest\n", test_x_scaled)
a = []
pred = []
for key, classifier in dict_classifiers.items():
    log_reg = classifier
    log_reg.fit(train_x_scaled, train_y)
    predictions = log_reg.predict(test_x_scaled)
    pred += [predictions]
    print(key + " accuracy score:")
    print(accuracy_score(test_y, predictions))
    a += [accuracy_score(test_y, predictions)]
    print("confusion matrix:")
    print(confusion_matrix(test_y, predictions))
    a += [confusion_matrix(test_y, predictions)]
    print("classification report:")
    print(classification_report(test_y, predictions))

print (pred )
# ROC curve et son aire sous la courbe appelée AUC
b = []
for key, classifier in dict_classifiers.items():
    y_pred_proba = classifier.predict_proba(test_x_scaled)[:,1]
    fpr, tpr, threshold = roc_curve(test_y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print( str(key) +' roc auc is:'+ str(roc_auc), '\n')
    b += [roc_auc]
    plt.plot(fpr,tpr, label= key)  #pb de labels
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()


# Les bénéfices réalisés par chaque classifieur (pas en return)
def benef(dic, data):
    c = []
    for key, classifier in dic.items():
        moula = 0
        cla = classifier
        cla.fit(train_x_scaled, train_y)
        predic = cla.predict(test_x_scaled)
        donnee = data['Close'][-len(predic) - 1:]
        for i in range (len(predic)):
            if predic[i] == 1:
                moula += donnee[i+1] - donnee[i]
        print(key +' : '+ str(moula)+'$')
        c += [moula]
    return c

c = benef( dict_classifiers, Btc)
print(c)


# création d'un tableau permettant de choisir les meilleurs classifieurs
clas = []
A =[]
B =[]
a = a[::-1]
b = b[::-1]
c = c[-6:]
c = c[::-1]
for key, classifier in dict_classifiers.items():
    clas += [[c.pop(), b.pop(), a.pop(), a.pop() ]]
    A += [key]
tab_cla = pd.DataFrame(clas, index= A,
                       columns= ['benefit', 'auc', 'accuracy score',
                                 'confusion matrix'])

pd.set_option('display.max_columns', None)
print('\n', tab_cla)


# nouvelle prédiction de la hausse avec la règle d'aggrégation 2/3
top_clas= pred[0]+ pred[4]+ pred[5]  #creer une fonction qui choisi d'elle-même les n meilleurs
final_clas = []
for k in range(len(top_clas)):
    if top_clas[k] >=2 :
        final_clas += [1]
    else:
        final_clas+= [0]
print(final_clas)
argent = 0
for i in range(len(final_clas)):
    if final_clas[i] == 1:
        argent+= Btc['Close'][-len(final_clas)+i+1] -Btc['Close'][-len(final_clas)+i]
print('benefice total' + ' : ' + str(argent) +'$')
