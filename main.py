import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print('Hello world')

bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates= True)
bitcoin.head()
bitcoin['Close'].plot()
plt.show()