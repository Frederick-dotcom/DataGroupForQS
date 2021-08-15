from influxdb import DataFrameClient
import pandas as pd
import json

host = 'ec2-3-23-98-149.us-east-2.compute.amazonaws.com'
port = 8086
username = 'cedric'
password = 'wasd123'
database = 'sb_local_db'
_14_03_21_params = {"_14_03_start":1615676400000000000, "_14_03_end":1615762799000000000}
_03_21lH = 'SELECT "speed"::field FROM "wind" WHERE TIME >= $_14_03_start AND TIME <= $_14_03_end '
_04_21_fH = 'SELECT "speed"::field FROM "wind" WHERE TIME >= 1615676400000000000 AND TIME <= 1618579525000000000 '

def createConnection(host, port, user, password, database):
    connection = DataFrameClient(host, port, user, password, database)
    print("Connection established!")
    return connection

def runQuery(client, query):
    print("running Query...")
    _04_21_fH_RS = client.query(_04_21_fH)
    print("Finished running query!")
    #print(_14_03_21RS)
    _04_21_fH_df = pd.DataFrame.from_dict(_04_21_fH_RS['wind'])
    print(_04_21_fH_df.head())
    print(_04_21_fH_df.tail())
    return _04_21_fH_df

def main():
    client = createConnection(host, port, username, password, database)
    _14_03 = runQuery(client, _04_21_fH)
    #_14_03.to_csv(r"C:\Users\juliu\Desktop\windSpeedCSVs\_16_03_bis_15_04_v2.csv", encoding='utf-8',
                 # na_rep='N/A', float_format='%.2f', columns=['speed'])
    #print("saved...")

if __name__ == "__main__":
    main()
