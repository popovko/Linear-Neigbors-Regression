import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

def f(x):
    return 2*x - 20

def prepare_df(X, y):
    df = pd.DataFrame(X, columns =  ['X'])
    df['y'] = y
    return df

#set default count points and disper
N = 50
disperce = 15

#set points and show it
delta = np.random.rand(N)[:, np.newaxis]*disperce
X = np.arange(N)[:, np.newaxis] + delta
y = f(np.arange(N)[:, np.newaxis]) - delta
data = prepare_df(X, y)
data.plot(kind = 'scatter', x = 'X', y = 'y', color = 'green')

#learn model

#model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5) neighbors alghoritm
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

#predict 2 points for bouid the line
test = [[0],[N+disperce]]
pred_test = model.predict(test)

#show result
plt.plot(test, pred_test, color = 'red')
plt.show()