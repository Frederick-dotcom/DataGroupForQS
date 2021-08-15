import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import pandas as pd
import scipy as sci

data_5min = '../Data/preFinal_5min.csv'
data_1min = '../Data/preFinal_1min.csv'

def readInData(path1, path2):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    return df1, df2

def plotData2(df_5min, df_1min):
    # Create figure and subplot manually
    # fig = plt.figure()
    # host = fig.add_subplot(111)

    # More versatile wrapper
    fig, host = plt.subplots(figsize=(15, 8))  # (width, height) in inches
    # (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)

    par1 = host.twinx()
    #par2 = host.twinx()

    #host.set_xlim(0, 50000)
    host.set_ylim(0, 35)
    par1.set_ylim(990, 1030)
    #par2.set_ylim(-180, 180)

    host.set_xlabel("Time")
    host.set_ylabel("Speed")
    par1.set_ylabel("Pressure")
    #par2.set_ylabel("Rel. Angle")

    color1 = plt.cm.plasma(0.2)
    color2 = plt.cm.inferno(0.5)
    color3 = plt.cm.viridis(.9)

    p1, = host.plot(df_1min['timestamp'], df_1min['speed'], color=color1, label="Speed")
    p2, = par1.plot(df_1min['timestamp'], df_1min['pressure'], color=color2, label="Pressure")
    #p3, = par2.plot(df_1min['timestamp'], df_1min['rel_angle'], color=color3, label="Rel. Angle")

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    #par2.spines['right'].set_position(('outward', 60))

    # no x-ticks
    #par2.xaxis.set_ticks([])

    # Sometimes handy, same for xaxis
    # par2.yaxis.set_ticks_position('right')

    # Move "Velocity"-axis to the left
    # par2.spines['left'].set_position(('outward', 60))
    # par2.spines['left'].set_visible(True)
    # par2.yaxis.set_label_position('left')
    # par2.yaxis.set_ticks_position('left')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    #par2.yaxis.label.set_color(p3.get_color())

    # Adjust spacings w.r.t. figsize
    fig.tight_layout()
    #plt.title("5min Agg.")
    plt.draw()
    plt.show()
    # Alternatively: bbox_inches='tight' within the plt.savefig function
    #                (overwrites figsize)


def plotData(df_5min, df_1min):
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    #host.set_xlim(0, 50000)
    host.set_ylim(0, 35)

    host.set_xlabel("Time")
    host.set_ylabel("Speed")
    par1.set_ylabel("Pressure")
    par2.set_ylabel("Rel. Angle")

    p1, = host.plot(df_1min['timestamp'], df_1min['speed'], label="Speed")
    p2, = par1.plot(df_1min['timestamp'], df_1min['pressure'], label="Pressure")
    p3, = par2.plot(df_1min['timestamp'], df_1min['rel_angle'], label="Rel. Angle")

    par1.set_ylim(990, 1030)
    par2.set_ylim(-180, 180)

    #host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    plt.draw()
    plt.show()

def main():
    df_5min, df_1min = readInData(data_5min, data_1min)
    df_1min['time'] = pd.to_datetime(df_1min['time'])
    df_5min['time'] = pd.to_datetime(df_5min['time'])
    df_5min['speed'].interpolate(inplace=True)
    df_5min['pressure'].interpolate(method="akima", inplace=True)
    df_5min['rel_angle'].interpolate(method="akima", inplace=True)
    df_1min['speed'].interpolate(inplace=True)
    df_1min['pressure'].interpolate(method="akima", inplace=True)
    df_1min['rel_angle'].interpolate(method="akima", inplace=True)
    #plotData(df_5min, df_1min)
    plotData2(df_5min, df_1min)

if __name__ == '__main__':
    main()
