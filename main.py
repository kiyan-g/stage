import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ta import add_all_ta_features
from sklearn.linear_model import LinearRegression

# Essai basic
print('Hello world')

# Figure bitcoin
bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates= True)
bitcoin.head()
bitcoin['Close'].plot()
plt.show()

# Essai rég linéaire
np.random.seed(0)
m = 100  #100 samples
x = np.linspace(0, 10, m).reshape(m, 1)
y = x + np.random.randn(m, 1)

plt.scatter(x, y)

model = LinearRegression()
model.fit(x, y)
model.score(x, y)
plt.show()