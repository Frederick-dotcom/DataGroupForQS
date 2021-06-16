import pandas as pd
import matplotlib.pyplot as plt

path1 = 'Data/windAngleData/windAngleMarch.csv'
path2 = 'Data/windAngleData/windAngleAprilfH.csv'
path3 = 'Data/windAngleData/windAngleAprilsH.csv'
path4 = 'Data/windAngleData/windAngleMaifH.csv'
path5 = 'Data/windAngleData/windAngleMai15_18.csv'

def readCSV(path1, path2, path3, path4, path5):
    print("reading in CSVs...")
    df1 = pd.read_csv(path1)
    print(df1.head())
    df2 = pd.read_csv(path2)
    print(df2.head())
    df3 = pd.read_csv(path3)
    print(df3.head())
    df4 = pd.read_csv(path4)
    print(df4.head())
    df5 = pd.read_csv(path5)
    print(df5.head())
    print("finished reading in CSVs")
    return df1, df2, df3, df4, df5

def putDFtogether(df1, df2, df3, df4, df5):
    pass
    df_list = [df1, df2, df3, df4, df5]
    df_final = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_final.head())
    return df_final

def convertToCSV(df_final):
    print("converting...")
    df_final.to_csv(r'C:\Users\juliu\Desktop\windAngleCSVs\csv_final.csv',  encoding='utf-8',
                  na_rep='N/A', float_format='%.2f', columns=['time', 'rel_angle'])
    print("saved...")

def main():
    df1, df2, df3, df4, df5 = readCSV(path1, path2, path3, path4, path5)
    df_final = putDFtogether(df1, df2, df3, df4, df5)
    convertToCSV(df_final)

if __name__ == "__main__":
    main()