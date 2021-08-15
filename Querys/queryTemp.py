from influxdb import DataFrameClient
import pandas as pd
import matplotlib.pyplot as plt

host = 'ec2-3-23-98-149.us-east-2.compute.amazonaws.com'
port = 8086
username = 'cedric'
password = 'wasd123'
database = 'sb_local_db'
QUERY = 'SELECT temperature::field FROM "water_temperature" WHERE TIME >= 1615849200000000000 '

def establishConnetion(host, port, username, password, database):
    client = DataFrameClient(host, port, username, password, database)
    print("connection established!")
    return client

def queryData(client, QUERY):
    print("running query...")
    rs = client.query(QUERY)
    print("finished running query!")
    return rs

def plotData(df):
    df.plot()
    plt.show()

def main():
    client = establishConnetion(host, port, username, password, database)
    #print(client.get_list_measurements())
    rs = queryData(client, QUERY)
    df = pd.DataFrame.from_dict(rs['water_temperature'])
    print(df.head())
    print("saving data...")
    df.to_csv(r'C:\Users\juliu\Desktop\waterTempCSVs\waterTemp.csv', encoding='utf-8',
                  na_rep='N/A', float_format='%.2f', columns=['temperature'])
    print("Save successful!")
    #plotData(df)

if __name__ == '__main__':
    main()