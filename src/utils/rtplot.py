import sys
sys.path.append("../")

import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np


class RealtimePlotData(object):  # added (by JadeCong)
    def __init__(self, size=(480, 640), title="Sim Data", xlabel="timestep", ylabel="value",
                 fix_data_range=False, data_range=2, range_buffer_size=100,
                 data_labels=['pos', 'ori', 'vel', 'force']):
        self.size = size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.fix_data_range = fix_data_range
        self.data_range = data_range
        self.range_buffer_size = range_buffer_size
        self.range_buffer = [0] * self.range_buffer_size

        self.data_labels = data_labels
        self.data_choice = len(self.data_labels)
        self.data_colors = ['red', 'green', 'blue', 'yellow', 'black', 'pink', 'orange', 'purple', 'brown', 'olive', 'cyan', 'gray', 'magenta']
        self.color_choice = len(self.data_colors)
        self.data_markers = ['o', '*', '+', 'p', '.', 'x', 'd', 'h', 's', 'v', '^', '<', '>']
        self.marker_choice = len(self.data_markers)

        self.fig = plt.figure()
        self.canvas = np.zeros(self.size)
        self.screen = pf.screen(self.canvas, self.title)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title, fontsize=10, color=self.data_colors[4])

    def plot_data(self, time, data):
        # get the data shape and data peak
        data_dim = data.shape[0]
        data_peak = max(data.min(), data.max(), key=abs)

        # dynamic data range for plotting
        del self.range_buffer[0]
        self.range_buffer.append(data_peak)
        range_buffer_peak = max(self.range_buffer)
        if self.fix_data_range:
            plt.ylim(-self.data_range, self.data_range)
        else:
            plt.ylim(-(range_buffer_peak + 1), range_buffer_peak + 1)
        plt.xlim(time[-1] - 5, time[-1])  # show the data of past five seconds

        # plot the data in lines
        line_list = []
        for idx in range(data_dim):
            if idx > self.color_choice or idx > self.marker_choice:  # the available colors
                raise IndexError
            line, = plt.plot(time, data[idx], color=self.data_colors[idx], marker=self.data_markers[idx], label=self.data_labels[idx], linewidth=1.5, markersize=3)
            line_list.append(line)
        plt.legend(handles=line_list, labels=self.data_labels[:data_dim], loc="upper right")  # set the legends for explaination

        # If we haven't already shown or saved the plot, then we need to draw the figure first...
        self.fig.canvas.draw()

        # update the screen
        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.screen.update(image)

    def end_plot(self):
        self.screen.close()
        print("End the plot and close the window...")
