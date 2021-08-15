import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import STL
from time import time
import seaborn as sn

#
# DataGroup
# Julius Schmidt, Cedric Riuz and Frederick Karallus
# 15.08.21
#


# Paths to our data
# stored in csv-files
PATH_1min = '../Data/preFinal_1min_v2.csv'
PATH_5min = '../Data/preFinal_5min_v2.csv'
PATH_20min = '../Data/preFinal_20min_v2.csv'
PATH_30min = '../Data/preFinal_30min_v2.csv'
PATH_1h = '../Data/preFinal_1h_v2.csv'

# Configuring titles for each Dataset
title_1min = '1min - Datensatz'
title_5min = '5min - Datensatz'
title_20min = '20min - Datensatz'
title_30min = '30min - Datensatz'
title_1h = '1h - Datensatz'

# Declare Frequencies
freq_1min = 'min'
freq_5min = '5min'
freq_20min = '20min'
freq_30min = '30min'
freq_1h = 'H'

# Declare window size for forecasting
# Always 60min divided by the dataset frequency
win_1min = 60
win_5min = 12
win_20min = 3
win_30min = 2
win_1h = 1

# Orders and Seasonal Orders for the SARIMAX models
# Was only being used in the beginning
# At the end only the "possible order" were being used
order_1min = (1, 1, 1)
s_order_1min = (1, 0, 1, int(win_1min*24))

order_5min = (1, 1, 1)
s_order_5min = (1, 1, 0, int(win_5min*4))

order_20min = (1, 1, 1)
s_order_20min = (1, 0, 1, int(win_20min*24))

order_30min = (1, 1, 1)
s_order_30min = (1, 1, 1, int(win_30min*24))

order_1h = (18, 1, 10)
s_order_1h = (1, 1, 1, int(win_1h*24))


# Possible orders
# As many orders as you like
poss_orders_5min = [(2, 1, 2)]
poss_s_orders_5min = [(2, 1, 2, int(win_5min*4))]

poss_orders_20min = [(0, 1, 0)]
poss_s_orders_20min = [(0, 1, 0, int(win_20min*24))]

poss_orders_30min = [(36, 1, 20)]
poss_s_orders_30min = [(1, 1, 1, int(win_30min*24))]

poss_orders_1h = [(10, 1, 10)]
poss_s_orders_1h = [(0, 1, 0, int(win_1h*24))]

#
# Import data
# Interpolate Data
# 'time' datatype --> datetime
# cleaning the datasets
#
def importData(PATH, period):
    # Reading in data
    df = pd.read_csv(PATH)

    # Interpolating data
    df_interpol = interpolate_Data(df)

    # Correcting datatype of 'time'-column
    df_interpol['time'] = pd.to_datetime(df_interpol['time'])

    # Converting 'time'-column to index
    df_interpol.index = df_interpol['time']

    # Removing unnecessary columns
    df_clean = remove_Unnecessary_Columns(df_interpol)

    #print(df_clean.head())

    # Returning DataFrame
    return df_clean

#
# Removing unnecessary columns from our dataframes
#
def remove_Unnecessary_Columns(df):
    try:
        del df['timestamp']
    except:
        print("No Column named: 'timestamp'")
    try:
        del df['Unnamed: 0']
    except:
        print("No Column named: 'Unnamed: 0'")
    try:
        del df['time']
    except:
        print("No Column named: 'time'")
    return df

#
# Function for interpolating DataFrames, mostly using 'akima' to fill
# Inplace = True
#
def interpolate_Data(df):
    df['speed'].interpolate(inplace=True)
    df['pressure'].interpolate(method="akima", inplace=True)
    df['pressure'].fillna(method='bfill', inplace=True)
    df['rel_angle'].interpolate(method="akima", inplace=True)
    df['temperature'].interpolate(method="akima", inplace=True)
    return df


# Function for checking correlations between variables in a DataFrame
def check_For_Correlation(df):
    corr = df.corr()
    return corr

# Function to visualize a correlation matrix
def vis_Corr(corr, title):
    sn.heatmap(corr, annot=True)
    plt.title(title)
    plt.show()


# Function for basic data visualization
def visData(df, title):

    fig, axs = plt.subplots(4, figsize=(15, 8))
    axs[0].plot(df['speed'], color='red', label='wind speed')
    axs[0].legend(loc='best')
    axs[1].plot(df['pressure'], color='blue', label='pressure')
    axs[1].legend(loc='best')
    axs[2].plot(df['rel_angle'], color='green', label='rel_angle')
    axs[2].legend(loc='best')
    axs[3].plot(df['temperature'], color='orange', label='temperature')
    axs[3].legend(loc='best')
    plt.suptitle("Raw Data View | " + title)
    # for days in range()
    plt.show()

#
# Indicates a daily trend
# A Dataframe is given and the function does a Seasonal Trend decomposition on each column
# The STl-results are then being visualized via the "visSTL"-function
#
def performingSLT(df, title, freq, win):

    slt = STL(df['speed'], period=win*24)
    results_speed = slt.fit()
    seasonal_speed, trend_speed, resid_speed = results_speed.seasonal, results_speed.trend, results_speed.resid
    visSTL(df['speed'], title, seasonal_speed, trend_speed, resid_speed, "Wind Speed")

    slt = STL(df['pressure'], period=win*24)
    results_pressure = slt.fit()
    seasonal_pressure, trend_pressure, resid_pressure = results_pressure.seasonal, results_pressure.trend,\
                                                        results_pressure.resid
    visSTL(df['pressure'], title, seasonal_pressure, trend_pressure, resid_pressure, "Pressure")

    slt = STL(df['rel_angle'], period=win*24)
    results_rel_angle = slt.fit()
    seasonal_rel_angle, trend_rel_angle, resid_rel_angle = results_rel_angle.seasonal, results_rel_angle.trend,\
                                                           results_rel_angle.resid
    visSTL(df['rel_angle'], title, seasonal_rel_angle, trend_rel_angle, resid_rel_angle, "rel_angle")

    slt = STL(df['temperature'], period=win*24)
    results_temperature = slt.fit()
    seasonal_temperature, trend_temperature, resid_temperature = results_temperature.seasonal, results_temperature.trend,\
                                               results_temperature.resid

    visSTL(df['temperature'], title, seasonal_temperature, trend_temperature, resid_temperature, "Temperature")


# Visualizing the STL-results
def visSTL(df, title, seasonal, trend, resid, i:str):

    fig, axs = plt.subplots(4, figsize=(15, 8))
    axs[0].plot(df, label='Data')
    axs[0].set_title('Data')
    axs[1].plot(seasonal, label='Seasonal')
    axs[1].set_title('Seasonal')
    axs[2].plot(trend, label='Trend')
    axs[2].set_title('Trend')
    axs[3].plot(resid, label='Residuals')
    axs[3].set_title('Residuals')
    plt.suptitle(i + " in " + title)
    fig.tight_layout()
    plt.show()

# Perform ADF-Test to check our data for stationarity
def adfuller_Test(df):
    for i in df.columns:
        print("ADF-Test for Column: " + i)
        dftest = adfuller(df[i], maxlag=100)
        adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# of Lags', '# of Observations'])

        for key, value in dftest[4].items():
            adf['Critical Value (%s)' % key] = value
        print(adf)

        p = adf['p-value']
        if p <= 0.05:
            print("\nSeries is Stationary")
        else:
            print("\nSeries is Non-Stationary")


# Help function, so that we could call just one function to test all dataframes at one time
# Needed to fill up some holes as well
# This might have happened due to the 'bfill' method
def perform_ADF(df_1min, df_5min, df_20min, df_30min, df_1h):
    print("-----------------------------------------------------------------------------------------------------------")
    print("DataFrame 1min Frequency")
    df_1min.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_1min.fillna(method='bfill', inplace=True)
    df_1min.fillna(method='ffill', inplace=True)
    try:
        adfuller_Test(df_1min)
    except:
        print("Inf or NaNs")

    print("-----------------------------------------------------------------------------------------------------------")
    print("DataFrame 5min Frequency")
    df_5min.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_5min.fillna(method='bfill', inplace=True)
    df_5min.fillna(method='ffill', inplace=True)
    try:
        adfuller_Test(df_5min)
    except:
        print("Inf or NaNs")

    print("-----------------------------------------------------------------------------------------------------------")
    print("DataFrame 20min Frequency")
    df_20min.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_20min.fillna(method='bfill', inplace=True)
    df_20min.fillna(method='ffill', inplace=True)
    try:
        adfuller_Test(df_20min)
    except:
        print("Inf or NaNs")

    print("-----------------------------------------------------------------------------------------------------------")
    print("DataFrame 30min Frequency")
    df_30min.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_30min.fillna(method='bfill', inplace=True)
    df_30min.fillna(method='ffill', inplace=True)
    try:
        adfuller_Test(df_30min)
    except:
        print("Inf or NaNs")

    print("-----------------------------------------------------------------------------------------------------------")
    print("DataFrame 1h Frequency")
    df_1h.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_1h.fillna(method='bfill', inplace=True)
    df_1h.fillna(method='ffill', inplace=True)
    try:
        adfuller_Test(df_1h)
    except:
        print("Inf or NaNs")


# Checks a given DataFrame for Auto-Correlation and Partial Auto-Correlation
def acf_pacf(df, title):
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    for i in df.columns:
        plot_acf(df[i], lags=100)
        plt.title(title + "- ACF for " + i)
        plot_pacf(df[i], lags=100)
        plt.title(title + " - PACF for " + i)
        plt.show()


# Transform a dataframe into a stationary dataframe
# By differencing the given dataframe
def get_stationarity(df, title):
    df_diff = df.copy()
    for i in df_diff.columns:
        df_diff[i] = df_diff[i].diff(periods=1)
        #df[i] = np.log10(df[i])
        # vis_Stationarity(df[i], i, title)
    return df_diff


# Visualizing Stationarity
# Basic function to visualize a dataframe
def vis_Stationarity(df, i, title):
    plt.plot(df)
    plt.title(i + " in " + title)
    plt.show()

# This function splits up the dataframes into train and test datasets
# Also the target variable and the independent variables are being split into two dataframes
def prepareDataSet(df, freq):
    df.fillna(method='bfill', inplace=True)
    df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Preparing the train datasets
    X_train = pd.DataFrame()
    X_train['pressure'] = df_train['pressure']
    X_train['rel_angle'] = df_train['rel_angle']
    X_train['temperature'] = df_train['temperature']
    X_train.index = pd.DatetimeIndex(df_train.index).to_period(freq)
    y_train = pd.DataFrame(df_train['speed'])
    y_train.index = pd.DatetimeIndex(df_train.index).to_period(freq)

    # Preparing the test datasets
    X_test = pd.DataFrame()
    X_test['pressure'] = df_test['pressure']
    X_test['rel_angle'] = df_test['rel_angle']
    X_test['temperature'] = df_test['temperature']
    X_test.index = pd.DatetimeIndex(df_test.index).to_period(freq)
    y_test = pd.DataFrame(df_test['speed'])
    y_test.index = pd.DatetimeIndex(df_test.index).to_period(freq)

    return X_train, y_train, X_test, y_test, df_train, df_test


# This function creates a SARIMAX-model based on a given dataframe as well as on a the given hyperparameters
# 'order' and 's_order' are the hyperparameters
# 'X_train' is being used as the exogenous variables and 'y_train' as the target variable
def createSARIMAX(df, title, freq, win, order, s_order):
    X_train, y_train, X_test, y_test, df_train, df_test = prepareDataSet(df, freq)

    print("-----------------------------------------------------------------------------------------------------------")
    print("SARIMAX Modell | " + title)
    print("Starting...")
    model = SARIMAX(endog=y_train.astype(float), exog=X_train.astype(float), order=order, seasonal_order=s_order, trend='c')
    print("Model is ready")
    start_time = time()
    print("Start Time Fit: " + str(start_time))
    results = model.fit()
    end_time = time()
    print("End Time Fit: " + str(end_time))
    est_time = end_time - start_time
    print("Time of the .fit(): " + str(est_time))

    return X_train, y_train, X_test, y_test, df_train, df_test, results, model

# Plotting the fitted values
def plotPred(df_pred, y_train, title):
    plt.title(title)
    plt.plot(y_train.values, color='blue', label='data', alpha=0.7)
    plt.plot(df_pred.values, color='red', label='prediction')
    plt.legend(loc='best')
    plt.show()

# Plotting a Forecast
# As well es the historic mean, max and min
def plotForecasts(forecast, df_test, y_test, X_test, title, window, mean, max, min):
    i = 0
    lastPeriod_mean = []
    lastPeriod_max = []
    lastPeriod_min = []
    while i < window:
        lastPeriod_mean.append(mean)
        lastPeriod_max.append(max)
        lastPeriod_min.append(min)
        i = i + 1

    plt.title("Forecast vs. Test Data | " + title + " | Steps: " + str(window))
    plt.plot(y_test.head(window).values, color='blue', label='data', alpha=0.7)
    plt.plot(forecast.values, color='red', label='forecast')

    #
    # these next three lines need to be uncommented to show the reference-model
    # plt.plot(lastPeriod_max, color='k', linestyle='--', label="max of historic window")
    # plt.plot(lastPeriod_mean, color='green', label='mean of historic window')
    # plt.plot(lastPeriod_min, color='k', linestyle='--', label='min of historic window')
    plt.legend(loc='best')
    plt.show()

    return lastPeriod_mean, lastPeriod_max, lastPeriod_min

# This function performs the actual forecast and also prints out the results
# It is always a 1h, 6h, 12h and 24h forecast being created and evaluated
def forecasting(results, y_test, X_test, df_test, window, title):
    print(title + " | " + "1h Forecast")
    forecast_1h = results.forecast(exog=X_test.head(window).values, steps=window, alpha=0.05)
    pd.DataFrame(forecast_1h)
    mse_forecast_1h = mean_squared_error(y_test.head(window).values, forecast_1h)
    mae_forecast_1h = mean_absolute_error(y_test.head(window).values, forecast_1h)
    mape_forecast_1h = mean_absolute_percentage_error(y_test.head(window).values, forecast_1h)
    rmse_forecast_1h = mean_squared_error(y_test.head(window).values, forecast_1h, squared=False)
    print("Mean Squared Error: " + str(mse_forecast_1h))
    print("Mean Absolut Error: " + str(mae_forecast_1h))
    print("Mean Absolut Percentage Error: " + str(mape_forecast_1h))
    print("Root Mean Squared Error: " + str(rmse_forecast_1h))
    print("-----------------------------------------------------------------------------------------------------------")


    print(title + " | " + "6h Forecast")
    forecast_6h = results.forecast(exog=X_test.head(window*6).values, steps=(window*6), alpha=0.05)
    pd.DataFrame(forecast_6h)
    mse_forecast_6h = mean_squared_error(y_test.head(window*6).values, forecast_6h)
    mae_forecast_6h = mean_absolute_error(y_test.head(window*6).values, forecast_6h)
    mape_forecast_6h = mean_absolute_percentage_error(y_test.head(window*6).values, forecast_6h)
    rmse_forecast_6h = mean_squared_error(y_test.head(window*6).values, forecast_6h, squared=False)
    print("Mean Squared Error: " + str(mse_forecast_6h))
    print("Mean Absolut Error: " + str(mae_forecast_6h))
    print("Mean Absolut Percentage Error: " + str(mape_forecast_6h))
    print("Root Mean Squared Error: " + str(rmse_forecast_6h))
    print("-----------------------------------------------------------------------------------------------------------")

    print(title + " | " + "12h Forecast")
    forecast_12h = results.forecast(exog=X_test.head(window*12).values, steps=(window*12), alpha=0.05)
    pd.DataFrame(forecast_12h)
    mse_forecast_12h = mean_squared_error(y_test.head(window*12).values, forecast_12h)
    mae_forecast_12h = mean_absolute_error(y_test.head(window * 12).values, forecast_12h)
    mape_forecast_12h = mean_absolute_percentage_error(y_test.head(window * 12).values, forecast_12h)
    rmse_forecast_12h = mean_squared_error(y_test.head(window * 12).values, forecast_12h, squared=False)
    print("Mean Squared Error: " + str(mse_forecast_12h))
    print("Mean Absolut Error: " + str(mae_forecast_12h))
    print("Mean Absolut Percentage Error: " + str(mape_forecast_12h))
    print("Root Mean Squared Error: " + str(rmse_forecast_12h))
    print("-----------------------------------------------------------------------------------------------------------")

    print(title + " | " + "24h Forecast")
    forecast_24h = results.forecast(exog=X_test.head(window*24).values, steps=(window*24), alpha=0.05)
    pd.DataFrame(forecast_24h)
    mse_forecast_24h = mean_squared_error(y_test.head(window*24).values, forecast_24h)
    mae_forecast_24h = mean_absolute_error(y_test.head(window * 24).values, forecast_24h)
    mape_forecast_24h = mean_absolute_percentage_error(y_test.head(window * 24).values, forecast_24h)
    rmse_forecast_24h = mean_squared_error(y_test.head(window * 24).values, forecast_24h, squared=False)
    print("Mean Squared Error: " + str(mse_forecast_24h))
    print("Mean Absolut Error: " + str(mae_forecast_24h))
    print("Mean Absolut Percentage Error: " + str(mape_forecast_24h))
    print("Root Mean Squared Error: " + str(rmse_forecast_24h))
    print("-----------------------------------------------------------------------------------------------------------")

    return mse_forecast_1h, mse_forecast_6h, mse_forecast_12h, mse_forecast_24h, forecast_1h, forecast_6h,\
           forecast_12h, forecast_24h

# help function to calculate the mean, max and min of a given period
def calc_mean_max_min(y_train, window, title):
    # 1h-calc
    mean_1h = y_train.tail(window).values.mean()
    max_1h = y_train.tail(window).values.max()
    min_1h = y_train.tail(window).values.min()
    print(title)
    print("Durchschnitt der letzten Stunde: " + str(mean_1h))
    print("Max. der letzten Stunde: " + str(max_1h))
    print("Min. der letzten Stunde: " + str(min_1h))

    # 6h-calc
    mean_6h = y_train.tail(window*6).values.mean()
    max_6h = y_train.tail(window*6).values.max()
    min_6h = y_train.tail(window*6).values.min()
    print(title)
    print("Durchschnitt der letzten 6 Stunden: " + str(mean_6h))
    print("Max. der letzten 6 Stunden: " + str(max_6h))
    print("Min. der letzten 6 Stunden: " + str(min_6h))

    # 12h-calc
    mean_12h = y_train.tail(window * 12).values.mean()
    max_12h = y_train.tail(window * 12).values.max()
    min_12h = y_train.tail(window * 12).values.min()
    print(title)
    print("Durchschnitt der letzten 12 Stunden: " + str(mean_12h))
    print("Max. der letzten 12 Stunden: " + str(max_12h))
    print("Min. der letzten 12 Stunden: " + str(min_12h))

    # 24h-calc
    mean_24h = y_train.tail(window * 24).values.mean()
    max_24h = y_train.tail(window * 24).values.max()
    min_24h = y_train.tail(window * 24).values.min()
    print(title)
    print("Durchschnitt der letzten 24 Stunden: " + str(mean_24h))
    print("Max. der letzten 24 Stunden: " + str(max_24h))
    print("Min. der letzten 24 Stunden: " + str(min_24h))

    return mean_1h, max_1h, min_1h, mean_6h, max_6h, min_6h, mean_12h, max_12h, min_12h, mean_24h, max_24h, min_24h

# Function to explore the results of a given model and its results
def exploreResults(model, results, X_train, y_train, X_test, y_test, df_train, df_test, df, title, window):

    # calculating the mean, max and min of a given 'y_train'-dataset
    mean_1h, max_1h, min_1h, mean_6h, max_6h, min_6h, mean_12h, max_12h, min_12h, mean_24h, max_24h, min_24h = \
        calc_mean_max_min(y_train, window, title)

    # printing out the results
    print(results.summary())
    resid = results.resid

    # Trying to fix the first residual value, due to a common statsmodels bug
    print(resid.iloc[0:1])
    resid.loc[0:1] = np.nan
    resid.fillna(method='bfill', inplace=True)
    print(resid.iloc[0:1])

    # printing out the residuals (mean, count etc.)
    print(resid.describe())

    # plotting the residuals over time
    plt.title("Residuals | " + title)
    plt.plot(resid.values)
    plt.show()

    # plotting the residuals distribution
    plt.title("Residuals Distribution | " + title)
    resid.plot(kind='kde')
    plt.show()

    # getting the prediction made on the results
    # basically the 'fitted values'
    pred = results.predict()
    df_pred = pd.DataFrame(pred)

    # Trying to fix the same bug like above
    print(df_pred['predicted_mean'].iloc[0:1])
    df_pred.loc[0:1, 'predicted_mean'] = np.nan
    df_pred.fillna(method='bfill', inplace=True)
    print(df_pred['predicted_mean'].iloc[0:1])

    # getting the MSE for fitte values as well as printing those out
    pred_mse = mean_squared_error(y_train, df_pred.values)
    print("Mean Squared Error of the fitted values " + title + " " + str(pred_mse))

    # Plotting the prediction (fitted values)
    plotPred(df_pred, y_train, title)

    # Performing the forecasts on a models results
    mse_forecast_1h, mse_forecast_6h, mse_forecast_12h, mse_forecast_24h, forecast_1h, forecast_6h, \
    forecast_12h, forecast_24h = forecasting(results, y_test, X_test, df_test, window, title)

    # Plotting the forecasts
    # Each forecast-period at a time
    lastPeriod_mean_1h, lastPeriod_max_1h, lastPeriod_min_1h  = \
        plotForecasts(forecast_1h, df_test, y_test, X_test, title, int(window), mean_1h, max_1h, min_1h)
    lastPeriod_mean_6h, lastPeriod_max_6h, lastPeriod_min_6h = \
        plotForecasts(forecast_6h, df_test, y_test, X_test, title, int(window*6), mean_6h, max_6h, min_6h)
    lastPeriod_mean_12h, lastPeriod_max_12h, lastPeriod_min_12h = \
        plotForecasts(forecast_12h, df_test, y_test, X_test, title, int(window*12), mean_12h, max_12h, min_12h)
    lastPeriod_mean_24h, lastPeriod_max_24h, lastPeriod_min_24h = \
        plotForecasts(forecast_24h, df_test, y_test, X_test, title, int(window*24), mean_24h, max_24h, min_24h)

    # calculating MSE, RMSE, MAE, MAPE for the reference-model
    mse_mean_1h = mean_squared_error(y_test.head(window).values, lastPeriod_mean_1h)
    mse_mean_6h = mean_squared_error(y_test.head(window*6).values, lastPeriod_mean_6h)
    mse_mean_12h = mean_squared_error(y_test.head(window*12).values, lastPeriod_mean_12h)
    mse_mean_24h = mean_squared_error(y_test.head(window*24).values, lastPeriod_mean_24h)

    rmse_mean_1h = mean_squared_error(y_test.head(window).values, lastPeriod_mean_1h, squared=False)
    rmse_mean_6h = mean_squared_error(y_test.head(window * 6).values, lastPeriod_mean_6h, squared=False)
    rmse_mean_12h = mean_squared_error(y_test.head(window * 12).values, lastPeriod_mean_12h, squared=False)
    rmse_mean_24h = mean_squared_error(y_test.head(window * 24).values, lastPeriod_mean_24h, squared=False)

    mae_mean_1h = mean_absolute_error(y_test.head(window).values, lastPeriod_mean_1h)
    mae_mean_6h = mean_absolute_error(y_test.head(window * 6).values, lastPeriod_mean_6h)
    mae_mean_12h = mean_absolute_error(y_test.head(window * 12).values, lastPeriod_mean_12h)
    mae_mean_24h = mean_absolute_error(y_test.head(window * 24).values, lastPeriod_mean_24h)

    mape_mean_1h = mean_absolute_percentage_error(y_test.head(window).values, lastPeriod_mean_1h)
    mape_mean_6h = mean_absolute_percentage_error(y_test.head(window * 6).values, lastPeriod_mean_6h)
    mape_mean_12h = mean_absolute_percentage_error(y_test.head(window * 12).values, lastPeriod_mean_12h)
    mape_mean_24h = mean_absolute_percentage_error(y_test.head(window * 24).values, lastPeriod_mean_24h)

    # printing out the results of the calculations made above
    print("----------------------------------------------------------------------------------------------------------")
    print("Referenzmodelle")
    print("MSE 1h Mean: " + str(mse_mean_1h) + " | MAE 1h Mean: " + str(mae_mean_1h) + " | RMSE 1h Mean: " +
          str(rmse_mean_1h) + " | MAPE 1h Mean: " + str(mape_mean_1h))
    print("MSE 6h Mean: " + str(mse_mean_6h) + " | MAE 6h Mean: " + str(mae_mean_6h) + " | RMSE 6h Mean: " +
          str(rmse_mean_6h) + " | MAPE 6h Mean: " + str(mape_mean_6h))
    print("MSE 12h Mean: " + str(mse_mean_12h) + " | MAE 12h Mean: " + str(mae_mean_12h) + " | RMSE 12h Mean: " +
          str(rmse_mean_12h) + " | MAPE 12h Mean: " + str(mape_mean_12h))
    print("MSE 24h Mean: " + str(mse_mean_24h) + " | MAE 24h Mean: " + str(mae_mean_24h) + " | RMSE 24h Mean: " +
          str(rmse_mean_24h) + " | MAPE 24h Mean: " + str(mape_mean_24h))
    print("----------------------------------------------------------------------------------------------------------")




# Main function
def main():

    # Importing data
    df_1min = importData(PATH_1min, freq_1min)
    df_5min = importData(PATH_5min, freq_5min)
    df_20min = importData(PATH_20min, freq_20min)
    df_30min = importData(PATH_30min, freq_30min)
    df_1h = importData(PATH_1h, freq_1h)

    # Visualizing the raw data
    visData(df_1min, title_1min)
    visData(df_5min, title_5min)
    visData(df_20min, title_20min)
    visData(df_30min, title_30min)
    visData(df_1h, title_1h)


    # Checking for Correlations between the input variables
    corr_1min = check_For_Correlation(df_1min)
    corr_5min = check_For_Correlation(df_5min)
    corr_20min = check_For_Correlation(df_20min)
    corr_30min = check_For_Correlation(df_30min)
    corr_1h = check_For_Correlation(df_1h)

    # Visualizing the Correlations
    vis_Corr(corr_1min, title_1min)
    vis_Corr(corr_5min, title_5min)
    vis_Corr(corr_20min, title_20min)
    vis_Corr(corr_30min, title_30min)
    vis_Corr(corr_1h, title_1h)

    # Perform Seasonal Trend Decomposition
    performingSLT(df_1min, title_1min, freq_1min, win_1min)
    performingSLT(df_5min, title_5min, freq_5min, win_5min)
    performingSLT(df_20min, title_20min, freq_20min, win_20min)
    performingSLT(df_30min, title_30min, freq_30min, win_30min)
    performingSLT(df_1h, title_1h, freq_1h, win_1h)

    # Check for stationarity
    perform_ADF(df_1min, df_5min, df_20min, df_30min, df_1h)

    # Making time series data stationary
    df_1min_diff = get_stationarity(df_1min, title_1min)
    df_5min_diff = get_stationarity(df_5min, title_5min)
    df_20min_diff = get_stationarity(df_20min, title_20min)
    df_30min_diff = get_stationarity(df_30min, title_30min)
    df_1h_diff = get_stationarity(df_1h, title_1h)

    # Perform ADF-Test on each differentiated DataFrame
    # One differentiation seems to be enough --> SAR(I == 1)MAX
    perform_ADF(df_1min_diff, df_5min_diff, df_20min_diff, df_30min_diff, df_1h_diff)

    # Checking ACF and PACF
    acf_pacf(df_1min_diff, title_1min)
    acf_pacf(df_5min_diff, title_5min)
    acf_pacf(df_20min_diff, title_20min)
    acf_pacf(df_30min_diff, title_30min)
    acf_pacf(df_1h_diff, title_1h)


    # Creating SARIMAX models and evaluating the results
    # We never used the 1min-Dataset due to its massive amount of needed computing power...
    # 1min-Freq - SARIMAX
    # X_train_1min, y_train_1min, X_test_1min, y_test_1min, df_train_1min, df_test_1min, results_1min, model_1min\
    #     = createSARIMAX(df_1min, title_1min, freq_1min, int(win_1min), order_1min, s_order_1min)

    # 1min-Freq - evaluating SARIMAX results
    # exploreResults(model_1min, results_1min, X_train_1min, y_train_1min, X_test_1min, y_test_1min,
    #                df_train_1min, df_test_1min, df_1min, title_1min, win_1min)


    # 20min-Freq - SARIMAX
    # Creates and evaluated each model based on the given poss_orders and poss_s_orders
    for order, s_order in zip(poss_orders_20min, poss_s_orders_20min):
        print("20min dataset")
        print("Order: " + str(order))
        print("S. Order: " + str(s_order))
        try:
            X_train_20min, y_train_20min, X_test_20min, y_test_20min, df_train_20min, df_test_20min, results_20min, model_20min\
                = createSARIMAX(df_20min, title_20min, freq_20min, int(win_20min * 24), order, s_order)

            # 20min-Freq - evaluating SARIMAX results
            exploreResults(model_20min, results_20min, X_train_20min, y_train_20min, X_test_20min, y_test_20min,
                           df_train_20min, df_test_20min, df_20min, title_20min, win_20min)
        except:
            print("Diese Iteration hatte wohl ein Problem mit der Order...")


    # 30min-Freq - SARIMAX
    for order, s_order in zip(poss_orders_30min, poss_s_orders_30min):
        print("30min dataset")
        print("Order: " + str(order))
        print("S. Order: " + str(s_order))
        try:
            X_train_30min, y_train_30min, X_test_30min, y_test_30min, df_train_30min, df_test_30min, results_30min, model_30min\
                = createSARIMAX(df_30min, title_30min, freq_30min, int(win_30min * 24), order, s_order)

            # 30min-Freq - evaluating SARIMAX results
            exploreResults(model_30min, results_30min, X_train_30min, y_train_30min, X_test_30min, y_test_30min,
                           df_train_30min, df_test_30min, df_30min, title_30min, win_30min)
        except:
            print("Diese Iteration hatte wohl ein Problem mit der Order...")


    # 60min-Freq - SARIMAX
    for order, s_order in zip(poss_orders_1h, poss_s_orders_1h):
        print("1h dataset")
        print("Order: " + str(order))
        print("S. Order: " + str(s_order))
        try:
            X_train_1h, y_train_1h, X_test_1h, y_test_1h, df_train_1h, df_test_1h, results_1h, model_1h\
                = createSARIMAX(df_1h, title_1h, freq_1h, int(win_1h * 24), order, s_order)

            # 60min-Freq - evaluating SARIMAX results
            exploreResults(model_1h, results_1h, X_train_1h, y_train_1h, X_test_1h, y_test_1h,
                           df_train_1h, df_test_1h, df_1h, title_1h, win_1h)
        except:
            print("Diese Iteration hatte wohl ein Problem mit der Order...")

    # 5min-Freq - SARIMAX
    for order, s_order in zip(poss_orders_5min, poss_s_orders_5min):
        print("5min dataset")
        print("Order: " + str(order))
        print("S. Order: " + str(s_order))
        try:
            X_train_5min, y_train_5min, X_test_5min, y_test_5min, df_train_5min, df_test_5min, results_5min, model_5min \
                = createSARIMAX(df_5min, title_5min, freq_5min, int(win_5min * 2), order, s_order)

            # 5min-Freq - evaluating SARIMAX results
            exploreResults(model_5min, results_5min, X_train_5min, y_train_5min, X_test_5min, y_test_5min,
                           df_train_5min, df_test_5min, df_5min, title_5min, win_5min)
        except:
            print("Diese Iteration hatte wohl ein Problem mit der Order...")


if __name__ == "__main__":
    main()