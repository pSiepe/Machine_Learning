#from itertools import product
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
# First we load the necesaary data

u_train_raw = np.load('/home/internet/Downloads/u_train.npy')
y_train_raw = np.load('/home/internet/Downloads/output_train.npy')
u_test_hand_in = np.load('/home/internet/Downloads/u_test.npy')


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

def mean_squared_error_trained(n,m,d,y_starting_values, u_data):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            x = compute_pred(n, m, d, y_starting_values, u_data)
    except ConvergenceWarning:
        return float('inf')
    return mean_squared_error(y_test[1:], x)

#print(train_regressor(2,2,3).coef_)

def compute_pred(n,m,d, y_starting_values, u_data):
    y_pred = y_starting_values[:get_max(n,m,d)]
    z = create_u_train(n,m,d,u_data)
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

#We will stick to alpha = 0.01 as aour hyperparameter for Lasso.

# Now we need to find the optimal tuple (n,m,d), which minimizes the mean squared error on our test data.
# Define the hyperparameter grid
n_values = [1, 2,3,4,5,6,7,8,9, 10]  # Example range for 'n'
m_values = [1,2,3,4, 5,6,7,8,9, 10]  # Example range for 'm'
d_values = [1, 2,3,4, 5,6,7,8,9,10]  # Example range for 'd'

# Store the best result
best_mse = float('inf')  # Start with infinity, to ensure any result is lower
best_hyperparameters = None  # To store the best (n, m, d) combination

# Iterate over all combinations of hyperparameters
'''for n, m, d in product(n_values, m_values, d_values):
    # Compute the MSE for the current combination
    mse = mean_squared_error_trained(n, m, d, y_test, u_test)
    
    # If this combination gives a lower MSE, update the best result
    if mse < best_mse:
        best_mse = mse
        best_hyperparameters = (n, m, d)

# Output the best hyperparameters and the corresponding MSE.

print(f'Best hyperparameters: n={best_hyperparameters[0]}, m={best_hyperparameters[1]}, d={best_hyperparameters[2]}')'''

# Let's test the code with our ideal values n=10,m=10,d=6 on our test data.

print(mean_squared_error(y_test[1:], compute_pred(1,10,3, y_test, u_test)))
plt.plot(range(len(y_test)-1), compute_pred(1,10,3, y_test, u_test), color = 'black')
plt.plot(range(len(y_test)-1), y_test[1:], color='red')
plt.show()

# Next we will create the predictions for u_test. For this remember: Testdaten knüpfen an die Trainingsdaten nahtlos an
# Deshalb soll man die letzten paar y werte der Trainingsdaten als Ausgang verwenden und dann die letzten 400 vorhergesagten y werte einreichen.

y_pred_hand_in = compute_pred(1,10,3, y_train_raw, u_test_hand_in)[-400:]
print(y_pred_hand_in)



