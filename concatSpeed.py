import pandas as pd
import matplotlib.pyplot as plt

path1 = 'Data/WindSpeedData/_16_03_bis_15_04.csv'
path2 = 'Data/WindSpeedData/_15_04sH.csv'
path3 = 'Data/WindSpeedData/_05_21.csv'

def readCSV(path1, path2, path3):
    print("reading in CSVs...")
    df1 = pd.read_csv(path1)
    print(df1.head())
    df2 = pd.read_csv(path2)
    print(df2.head())
    df3 = pd.read_csv(path3)
    print(df3.head())
    print("finished reading in CSVs")
    return df1, df2, df3

def putDFtogether(df1, df2, df3):
    pass
    df_list = [df1, df2, df3]
    df_final = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_final.head())
    return df_final

def convertToCSV(df_final):
    print("converting...")
    df_final.to_csv(r'C:\Users\juliu\Desktop\windSpeedCSVs\csv_final2.csv',  encoding='utf-8',
                  na_rep='N/A', float_format='%.2f', columns=['time', 'speed'])
    print("saved...")

def main():
    df1, df2, df3 = readCSV(path1, path2, path3)
    df_final = putDFtogether(df1, df2, df3)
    convertToCSV(df_final)

if __name__ == "__main__":
    main()