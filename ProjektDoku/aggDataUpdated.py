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

#Aggregating data via resample-function
def agg(speedDF, angleDF, pressureDF):
    print("Resampling...")
    #https://towardsdatascience.com/how-to-group-data-by-different-time-intervals-using-python-pandas-eb7134f9b9b0
    speed_agg = speedDF.resample('min', on='time').speed.mean()
    angle_agg = angleDF.resample('min', on='time').rel_angle.mean()
    pressure_agg = pressureDF.resample('min', on='time').pressure.mean()
    print("Returning resampled data!")
    return speed_agg, angle_agg, pressure_agg


def joinData(speed_agg_df, angle_agg_df, pressure_agg_df):
    #Merging the data. First merging the angle-data onto the speed-data. Afertwords merging the pressure-data onto both.
    print("Attempting to merge data")
    semiFinalDF = speed_agg_df.join(angle_agg_df, on='timestamp')
    finalDF = semiFinalDF.join(pressure_agg_df, on='timestamp')
    print("Returning merged data!")
    return finalDF

def fixingNaN():
    #for fixing NaN Values later on
    pass

def plotSpeedData(speed_agg_df, speed_df, speed_agg):
    #plotting the aggregated speed data as well as the standard dataframe
    fig, axs = plt.subplots(3)
    axs[0].plot(speed_agg_df)
    axs[1].plot(speed_df['speed'])
    axs[2].plot(speed_agg)
    plt.show()

def plotAngleData(angle_agg_df, angle_df, angle_agg):
    fig, axs = plt.subplots(3)
    axs[0].plot(angle_agg_df)
    axs[1].plot(angle_df['rel_angle'])
    axs[2].plot(angle_agg)
    plt.show()

def plotAllAggData(speed_agg_df, angle_agg_df, finalDF):
    fig, axs = plt.subplots(3)
    axs[0].plot(speed_agg_df['timestamp'], speed_agg_df['speed'])
    axs[1].plot(angle_agg_df['timestamp'], angle_agg_df['rel_angle'])
    axs[2].plot(finalDF)
    plt.show()

def toCSV(dataframe):
    #converting the dataframe into a csv file
    dataframe.to_csv(r'C:\Users\juliu\Desktop\windSpeedCSVs\windSpeedAgg1.csv')

def main():
    #Reading in data
    speedDF, angleDF, pressureDF = readInData(speedPath, anglePath, pressurePath)

    #Aggregating data via resample-method
    speed_agg, angle_agg, pressure_agg = agg(speedDF, angleDF, pressureDF)

    #Converting Pandas Series to DataFrame (Speed)
    speed_agg_df = speed_agg.to_frame()
    speed_agg_df['timestamp'] = pd.to_numeric(speed_agg_df.index)
    print(speed_agg_df)

    #Converting Pandas Series to DataFrame (Angle)
    angle_agg_df = angle_agg.to_frame()
    angle_agg_df['timestamp'] = pd.to_numeric(angle_agg_df.index)
    print(angle_agg_df)

    #Converting Pandas Series to DataFrame (Pressure)
    pressure_agg_df = pd.DataFrame(pressure_agg)
    pressure_agg_df['timestamp'] = pd.to_numeric(pressure_agg_df.index)
    print(pressure_agg_df)

    #Joining Dataframes
    finalDF = joinData(speed_agg_df, angle_agg_df, pressure_agg_df)
    print(finalDF)

    #Saving as CSV-File
    #toCSV(speed_agg_df)

    #Plotting Data
    #plotSpeedData(speed_agg_df, speedDF, speed_agg)
    #plotAngleData(angle_agg_df, angleDF, angle_agg)
    #plotAllAggData(speed_agg_df, angle_agg_df, finalDF)

if __name__ == "__main__":
    main()
