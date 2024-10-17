from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
# First we load the necesaary data

u_train_raw = np.load('/home/internet/Downloads/u_train.npy')
y_train_raw = np.load('/home/internet/Downloads/output_train.npy')
u_test = np.load('/home/internet/Downloads/u_test.npy')


def get_max(n,m,d):
    return max(n,m+d)


split_index = int(len(u_train_raw) * 0.8)
u_train, u_test = u_train_raw[:split_index], u_train_raw[split_index:]
y_train, y_test = y_train_raw[:split_index], y_train_raw[split_index:]


def create_y_train(n,m,d, train_var):
    df1 = pd.DataFrame()
    for i in range(1, n+1):
        df1[f'y_col_{i}'] = train_var[get_max(n,m,d)-i:len(train_var)-i]
    return df1

def create_u_train(n,m,d, train_var):
    df2 = pd.DataFrame()
    for i in range(0, m+1):
        df2[f'u_col_{i}'] = train_var[get_max(n,m,d)-d-i:len(train_var)-d-i]
    return df2

def train_regressor(n,m,d):
    training_data = pd.concat([create_u_train(n,m,d, u_train), create_y_train(n,m,d, y_train)], axis=1)
    training_data.columns = [f'Col{i + 1}' for i in range(len(training_data.columns))]
    #print(training_data)
    #print(y_train[get_max(n,m,d):])
    reg = Lasso(alpha=0.01).fit(training_data, y_train[get_max(n,m,d):])
    return reg

#print(train_regressor(2,2,3).coef_)
def compute_pred(n,m,d):
    y_pred = y_test[:get_max(n,m,d)]
    z = create_u_train(n,m,d,u_test)
    model = train_regressor(n,m,d)
    for k in range(1,len(z)):
        row = z.iloc[k]
        g = np.concatenate((row, y_pred[-n:]))
        #print(g)
        h = pd.DataFrame([g], columns=[f'Col{i + 1}' for i in range(n+m+1)])
        #print(h)
        new_y = model.predict(h)
        y_pred = np.append(y_pred, new_y)
    return y_pred

print(mean_squared_error(compute_pred(10,10,6), y_test[1:]))
plt.plot(range(len(y_test)-1), compute_pred(10,10,6), color = 'black')
plt.plot(range(len(y_test)-1), y_test[1:], color='red')
plt.show()

