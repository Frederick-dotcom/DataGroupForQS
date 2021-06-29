from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error

PATH_1min = '../Data/preFinal_1min_v2.csv'
PATH_5min = '../Data/preFinal_5min_v2.csv'
PATH_20min = '../Data/preFinal_20min_v2.csv'
PATH_30min = '../Data/preFinal_30min_v2.csv'
PATH_1h = '../Data/preFinal_1h_v2.csv'

def importData(PATH_1min, PATH_5min, PATH_20min, PATH_30min, PATH_1h):
    print("Reading in data...")

    df_1min = pd.read_csv(PATH_1min)
    df_5min = pd.read_csv(PATH_5min)
    df_20min = pd.read_csv(PATH_20min)
    df_30min = pd.read_csv(PATH_30min)
    df_1h = pd.read_csv(PATH_1h)

    print("Data extracted!")
    return df_1min, df_5min, df_20min, df_30min, df_1h

def interpol(df_1min, df_5min, df_20min, df_30min, df_1h):
    print("Interpolating")

    # Interpolating df_1min
    df_1min['speed'].interpolate(inplace=True)
    df_1min['pressure'].interpolate(method="akima", inplace=True)
    df_1min['pressure'].fillna(method='bfill', inplace=True)
    df_1min['rel_angle'].interpolate(method="akima", inplace=True)
    df_1min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_5min
    df_5min['speed'].interpolate(inplace=True)
    df_5min['pressure'].interpolate(method="akima", inplace=True)
    df_5min['pressure'].fillna(method='bfill', inplace=True)
    df_5min['rel_angle'].interpolate(method="akima", inplace=True)
    df_5min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_20min
    df_20min['speed'].interpolate(inplace=True)
    df_20min['pressure'].interpolate(method='akima', inplace=True)
    df_20min['pressure'].fillna(method='bfill', inplace=True)
    df_20min['rel_angle'].interpolate(method="akima", inplace=True)
    df_20min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_30min
    df_30min['speed'].interpolate(inplace=True)
    df_30min['pressure'].interpolate(method="akima", inplace=True)
    df_30min['pressure'].fillna(method='bfill', inplace=True)
    df_30min['rel_angle'].interpolate(method="akima", inplace=True)
    df_30min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_1h
    df_1h['speed'].interpolate(inplace=True)
    df_1h['pressure'].interpolate(method="akima", inplace=True)
    df_1h['pressure'].fillna(method='bfill', inplace=True)
    df_1h['rel_angle'].interpolate(method="akima", inplace=True)
    df_1h['temperature'].interpolate(method="akima", inplace=True)

    print("Interpolated successfully!")

    return df_1min, df_5min, df_20min, df_30min, df_1h

def performADF(df):
    dftest = adfuller(df, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# of Lags', '# of Observations'])

    for key, value in dftest[4].items():
        adf['Critical Value (%s)'%key] = value
    print(adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")


def differ(ds_train):
    ds_train_diff = ds_train.diff().dropna()
    return ds_train_diff

def modelVAR(ds_train_diff, n_obs, ds):
    model = VAR(ds_train_diff)
    results = model.fit(maxlags=15, ic='aic')
    print(results.summary())
    lag_order = results.k_ar
    predicted = results.forecast(ds_train_diff.values[int(-lag_order):], (len(ds.index) - int(n_obs)))
    del ds['time']
    forecast = pd.DataFrame(predicted, index=ds.index[int(n_obs):], columns=ds.columns)
    forecast['pressure'] = (ds['pressure'].iloc[-1] - ds['pressure'].iloc[-2]) + forecast['pressure'].cumsum()
    forecast['rel_angle'] = (ds['pressure'].iloc[-1] - ds['pressure'].iloc[-2]) + forecast['pressure'].cumsum()
    forecast['temperature'] = (ds['temperature'].iloc[-1] - ds['temperature'].iloc[-2]) + forecast['temperature'].cumsum()
    return forecast


def plotVAR(ds, forecast):
    fig, axs = plt.subplots(2)
    axs[0].plot(ds)
    axs[1].plot(forecast)
    plt.show()


def main():
    # Reading in data
    df_1min, df_5min, df_20min, df_30min, df_1h = importData(PATH_1min, PATH_5min, PATH_20min, PATH_30min, PATH_1h)

    # Interpolating missing values
    df_1min_f, df_5min_f, df_20min_f, df_30min_f, df_1h_f = interpol(df_1min, df_5min, df_20min, df_30min, df_1h)

    # Dropping time and timestamp columns as they do not matter. Evaluating observations.
    del df_1min_f['timestamp']
    del df_1min_f['Unnamed: 0']
    n_obs_1min = (len(df_1min_f.index))*0.7
    print("1min | N. of Obs.: " + str(n_obs_1min))

    del df_5min_f['timestamp']
    del df_5min_f['Unnamed: 0']
    n_obs_5min = (len(df_5min_f.index))*0.7
    print("5min | N. of Obs.: " + str(n_obs_5min))

    del df_20min_f['timestamp']
    n_obs_20min = (len(df_20min_f.index))*0.7
    print("20min | N. of Obs.: " + str(n_obs_20min))

    del df_30min_f['timestamp']
    n_obs_30min = (len(df_30min_f.index))*0.7
    print("30min | N. of Obs.: " + str(n_obs_30min))

    del df_1h_f['timestamp']
    n_obs_1h = (len(df_1h_f.index))*0.7
    print("1h | N. of Obs.: " + str(n_obs_1h))

    # Split data
    ds_1min_train, ds_1min_test = df_1min_f[:int(-n_obs_1min)], df_1min_f[int(-n_obs_1min):]
    ds_5min_train, ds_5min_test = df_5min_f[:int(-n_obs_5min)], df_5min_f[int(-n_obs_5min):]
    ds_20min_train, ds_20min_test = df_20min_f[:int(-n_obs_20min)], df_20min_f[int(-n_obs_20min):]
    ds_30min_train, ds_30min_test = df_30min_f[:int(-n_obs_30min)], df_30min_f[int(-n_obs_30min):]
    ds_1h_train, ds_1h_test = df_1h_f[:int(-n_obs_1h)], df_1h_f[int(-n_obs_1h):]

    ds_1min_time = ds_1min_train['time']
    del ds_1min_train['time']
    ds_5min_time = ds_5min_train['time']
    del ds_5min_train['time']
    ds_20min_time = ds_20min_train['time']
    del ds_20min_train['time']
    ds_30min_time = ds_30min_train['time']
    del ds_30min_train['time']
    ds_1h_time = ds_1h_train['time']
    del ds_1h_train['time']

    # Differencing variables, which are Non-Stationary
    speed_1min = pd.DataFrame(ds_1min_train['speed'])
    ds_1min_train_diff = differ(ds_1min_train)
    ds_1min_train_diff['speed'] = speed_1min
    #print(ds_1min_train_diff.head())

    speed_5min = pd.DataFrame(ds_5min_train['speed'])
    ds_5min_train_diff = differ(ds_5min_train)
    ds_5min_train_diff['speed'] = speed_5min
    #print(ds_5min_train_diff.head())

    speed_20min = pd.DataFrame(ds_20min_train['speed'])
    ds_20min_train_diff = differ(ds_20min_train)
    ds_20min_train_diff['speed'] = speed_20min
    #print(ds_20min_train_diff.head())

    speed_30min = pd.DataFrame(ds_30min_train['speed'])
    ds_30min_train_diff = differ(ds_30min_train)
    ds_30min_train_diff['speed'] = speed_30min
    #print(ds_30min_train_diff.head())

    speed_1h = pd.DataFrame(ds_1h_train['speed'])
    ds_1h_train_diff = differ(ds_1h_train)
    ds_1h_train_diff['speed'] = speed_1h
    #print(ds_1h_train_diff.head())

    # Perform ADF Test
    for i in ds_1min_train_diff.columns:
        print("1min | Column: ", i)
        print("------------------------------------")
        #performADF(ds_1min_train_diff[i])
        print('\n')

    for i in ds_5min_train_diff.columns:
        print("5min | Column: ", i)
        print("------------------------------------")
        #performADF(ds_1min_train_diff[i])
        print('\n')

    for i in ds_20min_train_diff.columns:
        print("20min | Column: ", i)
        print("------------------------------------")
        #performADF(ds_1min_train_diff[i])
        print('\n')

    for i in ds_30min_train_diff.columns:
        print("30min | Column: ", i)
        print("------------------------------------")
        #performADF(ds_1min_train_diff[i])
        print('\n')

    for i in ds_1h_train_diff.columns:
        print("1h | Column: ", i)
        print("------------------------------------")
        #performADF(ds_1min_train_diff[i])
        print('\n')

    # Reintegrating time

    ds_1min_train_diff['time'] = ds_1min_time
    ds_1min_train_diff['time'] = pd.to_datetime(ds_1min_train_diff['time'])
    #ds_1min_train_diff['time'] = pd.to_numeric(ds_1min_train_diff['time'])
    ds_1min_train_diff.index = pd.DatetimeIndex(ds_1min_train_diff['time']).to_period('min')
    del ds_1min_train_diff['time']
    print(ds_1min_train_diff.head())

    ds_5min_train_diff['time'] = ds_5min_time
    ds_5min_train_diff['time'] = pd.to_datetime(ds_5min_train_diff['time'])
    #ds_5min_train_diff['time'] = pd.to_numeric(ds_5min_train_diff['time'])
    ds_5min_train_diff.index = pd.DatetimeIndex(ds_5min_train_diff['time']).to_period('min')
    del ds_5min_train_diff['time']
    print(ds_5min_train_diff.head())

    ds_20min_train_diff['time'] = ds_20min_time
    ds_20min_train_diff['time'] = pd.to_datetime(ds_20min_train_diff['time'])
    #ds_20min_train_diff['time'] = pd.to_numeric(ds_20min_train_diff['time'])
    ds_20min_train_diff.index = pd.DatetimeIndex(ds_20min_train_diff['time']).to_period('min')
    del ds_20min_train_diff['time']
    print(ds_20min_train_diff.head())

    ds_30min_train_diff['time'] = ds_30min_time
    ds_30min_train_diff['time'] = pd.to_datetime(ds_30min_train_diff['time'])
    #ds_30min_train_diff['time'] = pd.to_numeric(ds_30min_train_diff['time'])
    ds_30min_train_diff.index = pd.DatetimeIndex(ds_30min_train_diff['time']).to_period('min')
    del ds_30min_train_diff['time']
    print(ds_30min_train_diff.head())

    ds_1h_train_diff['time'] = ds_1h_time
    ds_1h_train_diff['time'] = pd.to_datetime(ds_1h_train_diff['time'])
    #ds_1h_train_diff['time'] = pd.to_numeric(ds_1h_train_diff['time'])
    ds_1h_train_diff.index = pd.DatetimeIndex(ds_1h_train_diff['time']).to_period('min')
    del ds_1h_train_diff['time']
    print(ds_1h_train_diff.head())

    # Setting time as index in test datasets
    ds_30min_test.index = pd.DatetimeIndex(ds_30min_test['time']).to_period('min')

    # Fitting VAR-Model + Forecasting size of test-set
    #forecast = modelVAR(ds_1min_train_diff, n_obs_1min, df_1min_f)
    #print(forecast.head())
    #plotVAR(df_1min_f, forecast)

    # Fitting ARIMA-Model + Forecasting
    model = ARIMA(ds_30min_train_diff['speed'], order=(15, 0, 0))
    model_fit = model.fit()
    print(model_fit.summary())
    # Plotting residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())
    output = model_fit.forecast(steps=5)
    print(output)
    df_output = pd.DataFrame.from_dict(output)
    #df_output['predicted_mean'] = float(df_output['predicted_mean'])
    #print(df_output['predicted_mean'].dtype)

    print(df_output)
    print(ds_30min_test.head())

    fig, axs = plt.subplots(2)
    axs[0].plot(df_output['predicted_mean'])
    axs[1].plot(ds_30min_test['speed'].head())
    plt.show()



if __name__ == "__main__":
    main()