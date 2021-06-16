from influxdb import DataFrameClient
import pandas as pd
import matplotlib.pyplot as plt

host = 'ec2-3-23-98-149.us-east-2.compute.amazonaws.com'
port = 8086
username = 'cedric'
password = 'wasd123'
database = 'sb_local_db'
windAngleQuery = 'SELECT "rel_angle"::field FROM "wind" WHERE TIME >= 1621029600000000000 '

def createConnection(host, port, username, password, database):
    client = DataFrameClient(host, port, username, password, database)
    print("Connection established!")
    return client

def runQuery(client, query):
    print("running query...")
    windAngleRS = client.query(query)
    #print(windAngleRS)
    print("finished running query!")
    return windAngleRS

def toDF(rs):
    df = pd.DataFrame.from_dict(rs['wind'])
    print("converted to df")
    return df

def toExcel(df):
    df.to_csv(r'C:\Users\juliu\Desktop\windAngleCSVs\windAngleMai15_18.csv', encoding='utf-8',
                  na_rep='N/A', float_format='%.2f', columns=['rel_angle'])
    print("saved...")

def main():
    client = createConnection(host, port, username, password, database)
    windAngleRS = runQuery(client, windAngleQuery)
    df = toDF(windAngleRS)
    print(df.head())
    toExcel(df)


if __name__ == "__main__":
    main()