import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import os
import glob
os.chdir("/Users/cedric/Desktop/Pycharm Folder")

#windspeed data
dataset1 = pd.read_csv("_16_03_bis_15_04.csv", decimal = '.', sep = ',')
dataset2 = pd.read_csv("_15_04sH.csv", decimal = '.', sep = ',')
dataset3 = pd.read_csv("_05_21.csv", decimal = '.', sep = ',')
frames = [dataset1, dataset2, dataset3]
#pressure data
dataset4 = pd.read_csv("pressure2.csv", decimal = '.', sep = ',')

#aggregate data to one csv
#def aggregateData():
   # pass

#plot data
#def plotting():
fig, axs = plt.subplots(5)
axs[0].plot(dataset1['speed'])
axs[1].plot(dataset2['speed'])
axs[2].plot(dataset3['speed'])
axs[3].plot(dataset4['pressure'])
plt.show()
wind = pd.concat(frames)
print (wind)
wind1 = pd.DataFrame (wind)
axs[4].plot(wind1['speed'])

#main method

#def main():
  #  dataset1 = pd.read_csv("_16_03_bis_15_04.csv", decimal='.', sep=',')
   # dataset2 = pd.read_csv("_15_04sH.csv", decimal='.', sep=',')
   # dataset3 = pd.read_csv("_05_21.csv", decimal='.', sep=',')
   # dataset4 = pd.read_csv("pressure2.csv", decimal='.', sep=',')
   # plotting()

#if "__name__" == "__main__":
#    main()