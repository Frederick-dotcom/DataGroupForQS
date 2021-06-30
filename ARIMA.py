import pandas as pd
import numpy as np
import statsmodels
from skimage.filters.edges import d1
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from math import ceil

# Import der Daten und erste veränderung der Datenstruktur
def importData():
    Data = pd.read_csv('preFinal_1h_v2.csv', sep=',')
    print(Data.iloc[616])
    print(Data)
    Data['time'] = pd.to_datetime(Data['time'])
    # print(Data.dtypes)
    Data.set_index('time', inplace=True)
    return Data

def bereinigen(Data):
    Data['speed'].interpolate(inplace=True)
    return Data

def adfTest(Data):
    #Data = np.reshape(d1, newshape=-1)
    #print(Data)
    #result = adfuller(Data)
    #print("Test Statistic = {:.4f}".format(result[0]))
    #print("p-value = {:.4f}".format(result[2]))

    dftest = adfuller(Data['speed'], autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# of Lags', '# of Observations'])

    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")


def correlation(Data):
    plot_acf(Data['speed'], lags=100)
    plot_pacf(Data['speed'], lags=100)
    plt.show()

def plot(Data, y_pred_test_exp):
    #plt.plot(Data['speed'])
    #plt.plot(y_pred_test_exp)
    #plt.show()

    # plotting prediction and original data
    #fig, axs = plt.subplots(2)
    #axs[0].plot(Data['speed'])
    #axs[1].plot(y_pred_test_exp)
    #plt.show()

    plt.plot(Data['speed'], color="red")
    plt.plot(y_pred_test_exp, color="blue")
    plt.show()

    
def trainingModel(Data):
    # split into test and train
    end_train = ceil(len(Data['speed'])*0.8)
    Data_train = Data['speed'].iloc[:end_train]
    Data_test = Data['speed'].iloc[end_train:]

    # training model
    arima = SARIMAX(Data_train, order=(1, 0, 0)) # 0, 2, 19
    result = arima.fit()
    print(result.summary())

    # printing residues
    result.resid.plot()
    plt.title('Residuen')
    plt.show()

    # predicting next 5 values
    print('result.forecast...')
    print(result.forecast(1))

    # predicting with more output values
    model = SARIMAX(Data['speed'], order=(1, 0, 0))
    result_new = model.filter(result.params)
    y_pred_test = result_new.predict(start='2021-05-12 07:00:00+00:00')
    return y_pred_test

def evaluate(Data, y_pred_test_exp):
    mae = statsmodels.tools.eval_measures.meanabs(Data['speed'].iloc[615:-1], y_pred_test_exp, axis=0)
    print('mae...')
    print(mae)
    rmse = statsmodels.tools.eval_measures.rmse(Data['speed'].iloc[615:-1], y_pred_test_exp, axis=0)
    print('rmse...')
    print(rmse)


def main():
    Data = importData()
    Data = bereinigen(Data)
    print(Data)
    #plot(Data)
    #adfTest(Data)
    #correlation(Data)
    y_pred_test = trainingModel(Data)
    plot(Data, y_pred_test)
    evaluate(Data, y_pred_test)

if __name__ == '__main__':
    main()
