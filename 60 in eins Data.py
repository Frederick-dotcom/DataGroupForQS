import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Einlesen von Daten und bereinigen
def bereinigen():
    Data = pd.read_csv("_16_03_bis_15_04.csv", sep=',')
    dfData = pd.DataFrame(Data)
    #print(type(dfData))
    dfData['Datetime'] = pd.to_datetime(dfData['Unnamed: 0'])
    # dfData = dfData.set_index("Datetime")
    del dfData['Unnamed: 0']
    #print(dfData)
    return dfData

def meansBerechnen1(dfData):
    # genervt = dfData.rolling(window=60, min_periods=60, center=True).mean()
    # genervt = pd.cut(genervt["speed"], bins=1956) # durch diese zeile wird der wind speed von z.B. 17.5 zu 1750
    #print(genervt)
    dfData['interval'] = (dfData['speed'])
    print(dfData)
    #mitMeans = dfData.groupby(['interval']).mean(60)
    #mitMeans = pd.cut(dfData["speed"], bins=1956)
    mitMeans = dfData['interval'].rolling(window=60, min_periods=60, center=False).mean()
    #anderesMeans = dfData['interval'].rolling(window=60, min_periods=60, center=False).mean()
    print(mitMeans)
    return mitMeans

def meansBerechnen2(dfData):
    dfData['interval'] = (dfData['speed'])
    print(dfData)
    anderesMeans = dfData['interval'].rolling(window=300, min_periods=300, center=False).mean()
    return anderesMeans

def plotten(mitMeans, dfData, anderesMeans):
    fig, axs = plt.subplots(3)
    axs[0].plot(dfData['speed'])
    axs[1].plot(mitMeans)
    axs[2].plot(anderesMeans)
    #plt.plot(mitMeans)
    #plt.title("Wind Speed mit mean")
    #plt.show()
    #plt.plot(dfData['speed'])
    #plt.title("Wind Speed ohne mean")
    plt.show()


def main():
    dfData = bereinigen()
    #print(dfData)
    mitMeans = meansBerechnen1(dfData)
    anderesMeans = meansBerechnen2(dfData)
    plotten(mitMeans, dfData, anderesMeans)


if __name__ == "__main__":
    main()

    # DataMeans = dfData['speed'].rolling(window=60, min_periods=60, center=True).mean()
    # genervt = dfData.resample(60)
    
