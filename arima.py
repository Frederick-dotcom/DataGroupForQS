import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import meanabs
from statsmodels.tsa.statespace.sarimax import SARIMAX

PATH_1min = '../Data/preFinal_1min_v2.csv'
PATH_5min = '../Data/preFinal_5min_v2.csv'
PATH_20min = '../Data/preFinal_20min_v2.csv'
PATH_30min = '../Data/preFinal_30min_v2.csv'
PATH_1h = '../Data/preFinal_1h_v2.csv'

# Reading in data
def readInData(PATH):
    df = pd.read_csv(PATH)
    interpol(df)
    df['time'] = pd.to_datetime(df['time'])
    return df

# Interpolating
def interpol(df):
    df['speed'].interpolate(inplace=True)
    df['pressure'].interpolate(method="akima", inplace=True)
    df['pressure'].fillna(method='bfill', inplace=True)
    df['rel_angle'].interpolate(method="akima", inplace=True)
    df['temperature'].interpolate(method="akima", inplace=True)

# Erstellung des Rolling Mean
# War eigentlich zur weiteren Überprüfung der Stationarität gedacht
# Naja, ist eigentlich eine unwichtige Funktion...
def rollingMean(df, window_1h, window_6h):
    try:
        del df['Unnamed: 0']
    except:
        print("Deletion of Unnamed: 0 failed...")

    print(window_6h)
    # rolling statistics
    rolling_mean_1h = df.rolling(window=window_1h).mean()
    rolling_std_1h = df.rolling(window=window_1h).std()
    rolling_mean_6h = df.rolling(window=window_6h).mean()
    rolling_std_6h = df.rolling(window=window_6h).std()

    return rolling_mean_1h, rolling_std_1h, rolling_mean_6h, rolling_std_6h

# Dies war die erste Plotting-Funktion
# Wird jetzt eigentlich nicht mehr benutzt
# Kann ignoriert werden
def plotData(df, roll_mean_1h, roll_std_1h, roll_mean_6h, roll_std_6h, result, prediction, forecast):

    df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
    plt.plot(df_train['speed'].values, color="blue", label="Original")
    #plt.plot(df_test['speed'].values, color="blue", label="Original")
    #plt.plot(roll_mean_6h['speed'].values, color="red", label="Roll. Mean")
    #plt.plot(roll_std_6h['speed'].values, color="green", label="Roll. Std.")
    #plt.plot(result.fittedvalues.values, color="grey", label="Model")
    plt.plot(prediction.values, color="red", label="Prediction")
    #plt.plot(forecast.values, color="grey", label="Forecast")
    plt.legend(loc="best")
    plt.show()

# Den ADF-Test für einen übergebenen Datensatz machen
def adfTest(df):
    dftest = adfuller(df['speed'], maxlag=100)
    adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# of Lags', '# of Observations'])

    for key, value in dftest[4].items():
        adf['Critical Value (%s)' % key] = value
    print(adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")

# Die ACF oder PACF für einen übergebenen Datensatz plotten
def acf_pacf(df):
    print(df.head())
    plot_acf(df['speed'])
    plot_pacf(df['speed'])
    plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# Hiermit können verschiedene Forecasts gegen den Testdatensatz geplottet werden
# Diese Funktion ist allerdings ohne den Durchschnitt...
#-----------------------------------------------------------------------------------------------------------------------
def plotDifferentModels(df, forecast_fullTest, forecast_1h, forecast_6h, forecast_24h, window):
    df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)

    # Checking for same index - War nur zur Überprüfung, ob ich richtig gearbeitet habe - kann ignoriert werden
    #print("--------------------------------------------------------------")
    #print("Checking for same index...")
    #print("df_test vs. forecast_fullTest")
    #print(df_test.head(5))
    #print(forecast_fullTest.head(5))
    #print("--------------------------------------------------------------")
    #print("Checking for same index...")
    #print("df_test vs. forecast_1h")
    #print(df_test.head(window))
    #print(forecast_1h.head())
    #print("--------------------------------------------------------------")
    #print("Checking for same index...")
    #print("df_test vs. forecast_6h")
    #print(df_test.head(window*6))
    #print(forecast_6h.head())
    #print("--------------------------------------------------------------")
    #print("Checking for same index...")
    #print("df_test vs. forecast_24h")
    #print(df_test.head(window*24))
    #print(forecast_24h.head())
    #print("--------------------------------------------------------------")

    fig, axs = plt.subplots(4)
    # Plotting full Test-Set vs. Complete Forecast
    axs[0].plot(df_test.values, label="Full Test-Set", color="blue")
    axs[0].plot(forecast_fullTest.values, label="Full Forecast", color="red")
    plt.legend(loc="best")

    # Plotting 1h Test-Set vs. 1h Forecast
    axs[1].plot(df_test.head(window).values, label="1h True Values", color="blue")
    axs[1].plot(forecast_1h.values, label="1h Forecast", color="red")
    plt.legend(loc="best")

    # Plotting 6h Test-Set vs. 6h Forecast
    axs[2].plot(df_test.head(window*6).values, label="6h True Values", color="blue")
    axs[2].plot(forecast_6h.values, label="6h Forecast", color="red")
    plt.legend(loc="best")

    # Plotting 24h Test-Set vs. 24h Forecast
    axs[3].plot(df_test.head(window*24).values, label="24h True Values", color="blue")
    axs[3].plot(forecast_24h.values, label="24h Forecast", color="red")
    plt.legend(loc="best")

    plt.show()


#-----------------------------------------------------------------------------------------------------------------------
# Dies ist die alte Version des ARIMA-Modells, hier kann/konnte nur mit dem 1h-Datensatz gearbeitet werden
# Die nächste Funktion ist die besser :)
#-----------------------------------------------------------------------------------------------------------------------

def arima_model_1hDF(df, window):

    df_clone = pd.DataFrame(df)
    del df_clone['timestamp']
    del df_clone['temperature']
    del df_clone['rel_angle']
    del df_clone['pressure']
    df_clone['time'] = pd.to_datetime(df_clone['time'])
    df_clone.index = pd.DatetimeIndex(df_clone['time']).to_period('H')
    del df_clone['time']
    print(df_clone.head())

    # Splitting ds into training and test ds
    df_clone_train, df_clone_test = train_test_split(df_clone, shuffle=False, train_size=0.8)

    # Creating the model
    model = ARIMA(df_clone_train, order=(23, 1, 2))

    # Fitting the model
    result = model.fit()

    #printing out summary
    print(result.summary())

    #printing out and plotting residuals
    residuals = pd.DataFrame(result.resid)
    print(residuals.describe())
    #plt.plot(residuals.values)
    #plt.show()

    # printing out Mean Squared Error
    print("Mean Squared Error | Prediction (Sklearn): ", mean_squared_error(df_clone_train, result.fittedvalues.values))

    # Predicting values
    prediction = result.predict()
    #print(prediction.head())

    # ------------------------
    #print(df_clone_train.tail())
    #print(df_clone_test.head())
    #print(df_clone_test.tail())
    # ------------------------

    # Forecasting values
    forecast_fullTest = result.forecast(steps=len(df_clone_test.index), alpha=0.05)
    forecast_6h = result.forecast(steps=6, alpha=0.05)
    forecast_1h = result.forecast(steps=1, alpha=0.05)
    forecast_24h = result.forecast(steps=24, alpha=0.05)
    prediction_1h = result.predict(start='2021-05-12 07', end='2021-05-12 07')
    print("--------------------------------------------------------------")
    forecast_fullTest_mse = mean_squared_error(df_clone_test['speed'].values, forecast_fullTest)
    forecast_1h_mse = mean_squared_error(df_clone_test['speed'].head(1).values, forecast_1h)
    forecast_6h_mse = mean_squared_error(df_clone_test['speed'].head(6).values, forecast_6h)
    forecast_24h_mse = mean_squared_error(df_clone_test['speed'].head(24).values, forecast_24h)
    print("--------------------------------------------------------------")
    print("1h-DF | Full Forecast Test-Set | MSE: ", forecast_fullTest_mse)
    print("1h-DF | 1h Forecast Test-Set | MSE: ", forecast_1h_mse)
    print("1h-DF | 6h Forecast Test-Set | MSE: ", forecast_6h_mse)
    print("1h-DF | 24h Forecast Test-Set | MSE: ", forecast_24h_mse)
    print("--------------------------------------------------------------")

    return result, prediction, forecast_fullTest, forecast_6h, forecast_1h, forecast_24h

#-----------------------------------------------------------------------------------------------------------------------
#
# Neue Version des ARIMA-Modells
# Jetzt kann auch "objekt-orientiert gearbeitet werde
# Es muss nur immer das richtige "window" mitübergeben werden, um dann die richtigen Forecasts zu  erstellen
# Ich habe immer mit dem 1h-Window des jeweiligen DFs gearbeitet. Die Windows sind in der Main-Methode definiert.
#
#-----------------------------------------------------------------------------------------------------------------------

def arima_model_(df, window):
    print("---------------------------------------------------------------")
    print("Universal DF - ARIMA model")
    df_clone = pd.DataFrame(df)
    try:
        del df_clone['Unnamed: 0']
    except:
        print("Hat keine 'Unnamed: 0' Spalte...")

    try:
        del df_clone['timestamp']
        del df_clone['temperature']
        del df_clone['rel_angle']
        del df_clone['pressure']
    except:
        print("Hat eine der Spalten nicht...")
    #df_clone['time'] = pd.to_datetime(df_clone['time'])
    #df_clone.index = pd.DatetimeIndex(df_clone['time']).to_period('5min')
    #del df_clone['time']
    print(df_clone.head())

    # Splitting ds into training and test ds
    df_clone_train, df_clone_test = train_test_split(df_clone, shuffle=False, train_size=0.8)

    print(df_clone_train.head())

    # Creating the model
    model = ARIMA(df_clone_train, order=(22, 1, 22))

    # Fitting the model
    result = model.fit()

    # printing out summary
    print(result.summary())

    # printing out and plotting residuals
    residuals = pd.DataFrame(result.resid)
    print(residuals.describe())
    # plt.plot(residuals.values)
    # plt.show()

    # printing out Mean Squared Error
    print("Mean Squared Error | Prediction (Sklearn): ", mean_squared_error(df_clone_train, result.fittedvalues.values))

    # Predicting values
    prediction = result.predict()
    # print(prediction.head())

    # ------------------------
    # print(df_clone_train.tail())
    # print(df_clone_test.head())
    # print(df_clone_test.tail())
    # ------------------------

    # Forecasting values
    forecast_fullTest = result.forecast(steps=len(df_clone_test.index), alpha=0.05)
    forecast_6h = result.forecast(steps=window*6, alpha=0.05)
    forecast_1h = result.forecast(steps=window*1, alpha=0.05)
    forecast_24h = result.forecast(steps=window*24, alpha=0.05)
    prediction_1h = result.predict(start='2021-05-12 07', end='2021-05-12 07')
    print("--------------------------------------------------------------")
    forecast_fullTest_mse = mean_squared_error(df_clone_test['speed'].values, forecast_fullTest)
    forecast_1h_mse = mean_squared_error(df_clone_test['speed'].head(window*1).values, forecast_1h)
    forecast_6h_mse = mean_squared_error(df_clone_test['speed'].head(window*6).values, forecast_6h)
    forecast_24h_mse = mean_squared_error(df_clone_test['speed'].head(window*24).values, forecast_24h)
    print("--------------------------------------------------------------")
    print("1min-DF | Full Forecast Test-Set | MSE: ", forecast_fullTest_mse)
    print("1min-DF | 1h Forecast Test-Set | MSE: ", forecast_1h_mse)
    print("1min-DF | 6h Forecast Test-Set | MSE: ", forecast_6h_mse)
    print("1min-DF | 24h Forecast Test-Set | MSE: ", forecast_24h_mse)
    print("--------------------------------------------------------------")

    return result, prediction, forecast_fullTest, forecast_6h, forecast_1h, forecast_24h

# Hiermit kann ein Forecast gegen seinen Testdatensatz geplottet werden
# Der Mean wird bislang noch "von Hand" gemacht xD
# Ist noch Verbesserungspotential
# Aber ich war spät dran ;)

def plotBestForecast(df, forecast_fullTest, forecast_1h, forecast_6h, forecast_24h):
    df_train, df_test = train_test_split(df, train_size=0.8, shuffle=False)
    mean = df_train['speed'].tail(24).mean()
    #print(mean)
    df_mean = pd.DataFrame([mean, mean, mean, mean, mean, mean, mean, mean, mean, mean, mean, mean,
                            mean, mean, mean, mean, mean, mean, mean, mean, mean, mean, mean, mean])
    plt.plot(df_test['speed'].head(24).values, label="Original", color="blue")
    plt.plot(df_mean.values, label="Ref. Modell", color="orange")
    plt.plot(forecast_24h.values, label="forecast", color="red")
    plt.legend(loc="best")
    plt.show()
    print(mean_squared_error(df_test.head(24).values, df_mean.values))

# Main Method
def main():
    # Einlesen der Daten und das definieren von Zeitfenstern
    # Diese Zeitfenster sind im Endeffekt nur Konstanten für die Anzahl der Lags,
    # die für einen gewissen Zeitraum benötigt wird
    df_1min = readInData(PATH_1min)
    window_1h_1min = 60
    window_6h_1min = 360
    df_5min = readInData(PATH_5min)
    window_1h_5min = 12
    window_6h_5min = 72
    df_20min = readInData(PATH_20min)
    window_1h_20min = 3
    window_6h_20min = 18
    df_30min = readInData(PATH_30min)
    window_1h_30min = 2
    window_6h_30min = 12
    df_1h = readInData(PATH_1h)
    window_1h_1h = 1
    window_6h_1h = 6

    # Rolling Mean erstellen, wird aber eig nicht mehr benötigt...
    roll_mean_1h, roll_std_1h, roll_mean_6h, roll_std_6h = rollingMean(df_1h, window_1h_1h, window_6h_1h)

    # ADF-Test machen
    #adfTest(df_1h)

    # ACF und PACF testen
    #acf_pacf(df_1min)

    # Hiermit habe ich immer den 1h-Datensatz in ein ARIMA gehauen. Eigentlich jetzt irrelevant
    result_1hDF, prediction_1hDF, forecast_fullTest_1hDF, forecast_6h_1hDF, forecast_1h_1hDF, forecast_24h_1hDF = \
        arima_model_1hDF(df_1h, window_1h_1h)

    # Das ist die "neue" ARIMA-Funktion, die funktioniert auch ganz gut eig
    result_DF, prediction_DF, forecast_fullTest_DF, forecast_6h_DF, forecast_1h_DF, forecast_24h_DF = \
        arima_model_(df_1h, window_1h_1h)

    # War zum Plotten von Datensätzen und den Forecast über verschiedene Zeiträume
    # Funktioniert grundsätzlich, aber eigentlich irrelevant
    #plotDifferentModels(df_1h, forecast_fullTest_1hDF, forecast_1h_1hDF, forecast_6h_1hDF, forecast_24h_1hDF,
     #                   window_1h_1h)
    #plotDifferentModels(df_5min, forecast_fullTest_DF, forecast_1h_DF, forecast_6h_DF, forecast_24h_DF,
    #                    window_1h_5min)

    # Alte Funktion, muss nicht verwendet werden...
    #plotData(df_1h, roll_mean_1h, roll_std_1h, roll_mean_6h, roll_std_6h, result_DF, prediction_DF,
    # forecast_fullTest_DF)

    # Dies ist die Wichtigste Plotting-Funktion, hiermit habe ich auch das einfache Referenzmodell geplottet
    plotBestForecast(df_1h, forecast_fullTest_DF, forecast_1h_DF, forecast_6h_DF, forecast_24h_DF)


if __name__ == "__main__":
    main()