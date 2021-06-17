import pandas as pd
import matplotlib.pyplot as plt

T_AGG = 60
speedPath = 'Data/WindSpeedData/csv_final2.csv'
anglePath = 'Data/windAngleData/csv_final.csv'
pressurePath = 'Data/PressureData/pressure2.csv'

#Reading in data as well as converting String-Objects into Datetimes
def readInData(speedPath, anglePath, pressurePath):
    print("Reading in data...")
    #Reading in speed data
    speedDF = pd.read_csv(speedPath)
    #converting object into datetime
    speedDF['time'] = pd.to_datetime(speedDF['time'])
    #Reading in angle data
    angleDF = pd.read_csv(anglePath)
    #converting object into datetime
    angleDF['time'] = pd.to_datetime(angleDF['time'])
    #Reading in pressure data
    pressureDF = pd.read_csv(pressurePath)
    #Converting object into datetime
    pressureDF['time'] = pd.to_datetime(pressureDF['time'])
    #Returning Dataframes
    print("Returning Dataframes!")
    return speedDF, angleDF, pressureDF

def agg(speedDF, angleDF, pressureDF):
    print("Resampling...")
    #https://towardsdatascience.com/how-to-group-data-by-different-time-intervals-using-python-pandas-eb7134f9b9b0
    speed_agg = speedDF.resample('min', on='time').speed.mean()
    angle_agg = angleDF.resample('min', on='time').rel_angle.mean()
    pressure_agg = pressureDF.resample('min', on='time').pressure.mean()
    print("Returning resampled data!")
    return speed_agg, angle_agg, pressure_agg

def joinData(speed_agg, angle_agg, pressure_agg):
    #Merging the data. First merging the angle-data onto the speed-data. Afertwords merging the pressure-data onto both.
    print("Attempting to merge data")
    finalDF = speed_agg.join(angle_agg, on='time').join(pressure_agg, on='time')
    print("Returning merged data!")
    return finalDF

def fixingNaN():
    #for fixing NaN Values later on
    pass

def plotSpeedData(speed_agg_df, speed_df):
    #plotting the aggregated speed data as well as the standard dataframe
    fig, axs = plt.subplots(2)
    axs[0].plot(speed_agg_df)
    axs[1].plot()

def toCSV(dataframe):
    #converting the dataframe into a csv file
    dataframe.to_csv(r'C:\Users\juliu\Desktop\aggCSV1.csv')

def main():
    speedDF, angleDF, pressureDF = readInData(speedPath, anglePath, pressurePath)
    speed_agg, angle_agg, pressure_agg = agg(speedDF, angleDF, pressureDF)
    speed_agg_df = pd.DataFrame(speed_agg)
    angle_agg_df = pd.DataFrame(angle_agg)
    pressure_agg_df = pd.DataFrame(pressure_agg)
    finalDF = joinData(speed_agg_df, angle_agg_df, pressure_agg_df)
    print(finalDF)
    #toCSV(finalDF)


if __name__ == "__main__":
    main()
