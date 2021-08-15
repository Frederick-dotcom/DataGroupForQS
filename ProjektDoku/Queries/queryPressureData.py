from influxdb import DataFrameClient
import pandas as pd
import matplotlib.pyplot as plt

host = 'ec2-3-23-98-149.us-east-2.compute.amazonaws.com'
port = 8086
username = 'cedric'
password = 'wasd123'
database = 'sb_local_db'
pressureQuery = 'SELECT "pressure"::field FROM "atmospheric_pressure" WHERE TIME >= 1615849200000000000 '

def createConnection(host, port, username, password, database):
    client = DataFrameClient(host, port, username, password, database)
    print("Connection established!")
    return client

def runQuery(client, query):
    print("running query...")
    pressureQueryRS = client.query(query)
    print("finished running query!")
    return pressureQueryRS

def toDF(rs):
    df = pd.DataFrame.from_dict(rs['atmospheric_pressure'])
    print("converted to df")
    return df

def toExcel(df):
    df.to_csv(r'C:\Users\juliu\Desktop\atmospheric_pressureCSVs\pressure2.csv', encoding='utf-8',
                  na_rep='N/A', float_format='%.2f', columns=['pressure'])
    print("saved...")

def main():
    client = createConnection(host, port, username, password, database)
    pressureQueryRS = runQuery(client, pressureQuery)
    df = toDF(pressureQueryRS)
    print(df.head())
    toExcel(df)


if __name__ == "__main__":
    main()
