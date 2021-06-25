import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from statsmodels.tsa.stattools import adfuller


# https://www.influxdata.com/blog/autocorrelation-in-time-series-data/
# https://towardsdatascience.com/detecting-stationarity-in-time-series-data-d29e0a21e638
# https://machinelearningmastery.com/time-series-data-stationary-python/


PATH_1min = '../Data/preFinal_1min_v2.csv'
PATH_5min = '../Data/preFinal_5min_v2.csv'
PATH_20min = '../Data/preFinal_20min_v2.csv'
PATH_30min = '../Data/preFinal_30min_v2.csv'
PATH_1h = '../Data/preFinal_1h_v2.csv'

# Reading in data
def readInData(PATH_1min, PATH_5min, PATH_20min, PATH_30min, PATH_1h):
    print("Reading in data...")

    df_1min = pd.read_csv(PATH_1min)
    df_5min = pd.read_csv(PATH_5min)
    df_20min = pd.read_csv(PATH_20min)
    df_30min = pd.read_csv(PATH_30min)
    df_1h = pd.read_csv(PATH_1h)

    print("Data extracted!")
    return df_1min, df_5min, df_20min, df_30min, df_1h

# Interpolating dataframes
def interpol(df_1min, df_5min, df_20min, df_30min, df_1h):
    print("Interpolating")

    # Interpolating df_1min
    df_1min['speed'].interpolate(inplace=True)
    df_1min['pressure'].interpolate(method="akima", inplace=True)
    df_1min['rel_angle'].interpolate(method="akima", inplace=True)
    df_1min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_5min
    df_5min['speed'].interpolate(inplace=True)
    df_5min['pressure'].interpolate(method="akima", inplace=True)
    df_5min['rel_angle'].interpolate(method="akima", inplace=True)
    df_5min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_20min
    df_20min['speed'].interpolate(inplace=True)
    df_20min['pressure'].interpolate(method="akima", inplace=True)
    df_20min['rel_angle'].interpolate(method="akima", inplace=True)
    df_20min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_30min
    df_30min['speed'].interpolate(inplace=True)
    df_30min['pressure'].interpolate(method="akima", inplace=True)
    df_30min['rel_angle'].interpolate(method="akima", inplace=True)
    df_30min['temperature'].interpolate(method="akima", inplace=True)

    # Interpolating df_1h
    df_1h['speed'].interpolate(inplace=True)
    df_1h['pressure'].interpolate(method="akima", inplace=True)
    df_1h['rel_angle'].interpolate(method="akima", inplace=True)
    df_1h['temperature'].interpolate(method="akima", inplace=True)

    print("Interpolated successfully!")

    return df_1min, df_5min, df_20min, df_30min, df_1h

# Plotting ACF
def checkACF(df_5min, df_30min, df_1h):
    print("Plotting data...")

    plot_acf(df_30min['speed'], lags=100)
    plot_pacf(df_30min['speed'], lags=100)
    plot_acf(df_30min['pressure'], lags=100)
    plot_acf(df_30min['rel_angle'], lags=100)
    plot_acf(df_30min['temperature'], lags=100)

    plt.show()
    print("Plotted data!")

# Checking ADF
def checkADF(df_5min, df_30min, df_1h):
    result_5min_speed = adfuller(df_5min['speed'])
    result_5min_pressure = adfuller(df_5min['pressure'])
    result_5min_relangle = adfuller(df_5min['rel_angle'])
    result_5min_temperature = adfuller(df_5min['temperature'])

    result_30min_speed = adfuller(df_30min['speed'])
    result_30min_pressure = adfuller(df_30min['pressure'])
    result_30min_relangle = adfuller(df_30min['rel_angle'])
    result_30min_temperature = adfuller(df_30min['temperature'])

    result_1h_speed = adfuller(df_1h['speed'])
    result_1h_pressure = adfuller(df_1h['pressure'])
    result_1h_relangle = adfuller(df_1h['rel_angle'])
    result_1h_temperature = adfuller(df_1h['temperature'])
    print("---------------------------------------------------------------------")
    print("5min | Speed | ADF Statistic: " + str(result_5min_speed[0]))
    print("5min | Speed| p-value: " + str(result_5min_speed[1]))
    print("5min | Pressure | ADF Statistic: " + str(result_5min_pressure[0]))
    print("5min | Pressure | p-value: " + str(result_5min_pressure[1]))
    print("5min | Rel. Angle | ADF Statistic: " + str(result_5min_relangle[0]))
    print("5min | Rel. Angle | p-value: " + str(result_5min_relangle[1]))
    print("5min | Temperatur | ADF Statistic: " + str(result_5min_temperature[0]))
    print("5min | Temperatur | p-value: " + str(result_5min_temperature[1]))
    print("---------------------------------------------------------------------")
    print("30min | Speed | ADF Statistic: " + str(result_30min_speed[0]))
    print("30min | Speed| p-value: " + str(result_30min_speed[1]))
    print("30min | Pressure | ADF Statistic: " + str(result_30min_pressure[0]))
    print("30min | Pressure | p-value: " + str(result_30min_pressure[1]))
    print("30min | Rel. Angle | ADF Statistic: " + str(result_30min_relangle[0]))
    print("30min | Rel. Angle | p-value: " + str(result_30min_relangle[1]))
    print("30min | Temperatur | ADF Statistic: " + str(result_30min_temperature[0]))
    print("30min | Temperatur | p-value: " + str(result_30min_temperature[1]))
    print("---------------------------------------------------------------------")
    print("1h | Speed | ADF Statistic: " + str(result_1h_speed[0]))
    print("1h | Speed| p-value: " + str(result_1h_speed[1]))
    print("1h | Pressure | ADF Statistic: " + str(result_1h_pressure[0]))
    print("1h | Pressure | p-value: " + str(result_1h_pressure[1]))
    print("1h | Rel. Angle | ADF Statistic: " + str(result_1h_relangle[0]))
    print("1h | Rel. Angle | p-value: " + str(result_1h_relangle[1]))
    print("1h | Temperatur | ADF Statistic: " + str(result_1h_temperature[0]))
    print("1h | Temperatur | p-value: " + str(result_1h_temperature[1]))
    print("---------------------------------------------------------------------")

# Main Function
def main():
    # Reading in data
    df_1min, df_5min, df_20min, df_30min, df_1h = readInData(PATH_1min, PATH_5min, PATH_20min, PATH_30min, PATH_1h)

    # Fixing 'time' columns
    print("Fixing columns...")

    df_1min['time'] = pd.to_datetime(df_1min['time'])
    df_5min['time'] = pd.to_datetime(df_5min['time'])
    df_20min['time'] = pd.to_datetime(df_20min['time'])
    df_30min['time'] = pd.to_datetime(df_30min['time'])
    df_1h['time'] = pd.to_datetime(df_1h['time'])

    print("Columns fixed!")

    # Interpolating data
    interpol(df_1min, df_5min, df_20min, df_30min, df_1h)

    # Checking ACF
    checkACF(df_5min, df_30min, df_1h)

    # Checking Augmented Dickey-Fuller Test
    checkADF(df_5min, df_30min, df_1h)

if __name__ == "__main__":
    main()


