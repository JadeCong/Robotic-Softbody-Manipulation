import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


class RealtimePlotWindow(object):  # by JadeCong
    def __init__(self, win_title="Sim Data", win_size=(1000, 600), win_foreground="w", win_background="b", antialias=True, useOpenGL=True,
                 xlabel="Timestep", ylabel=["Position", "Orientation", "Velocity", "Force", "Derivative_Force"], plot_markersize=0.1, plot_linewidth=1.5,
                 data_labels=["pos", "ori", "vel", "force", "der_force"], data_references=[0.04, 5, 0], data_standard_unit=True, data_buffer_size=100, data_interval=1,
                 update_frequency=50, plot_together=True):
        # initialize the parameters
        self.win_title = win_title
        self.win_size = win_size
        self.win_foreground = win_foreground
        self.win_background = win_background
        self.antialias = antialias
        self.useOpenGL = useOpenGL
        self.xlabel = xlabel
        self.xlabel_units = ['s', 'ms']  # maybe add the timestep(ts)
        self.ylabel = ylabel
        self.ylabel_units = {'Position': ['m', 'mm'], 'Orientation': ['rad', 'deg'], 'Velocity': ['m/s', 'mm/s', 'rad/s', 'deg/s'], 'Force': ['N', 'mN', 'N.m', 'mN.m'], 'Derivative_Force': ['N/s', 'mN/s']}
        self.plot_markersize = plot_markersize
        self.plot_linewidth = plot_linewidth
        self.data_labels = data_labels
        self.data_choice = len(self.data_labels)
        self.data_colors = ['r', 'g', 'b', 'y', 'k', 'p', 'o', 'c', 'g', 'm']
        self.color_choice = len(self.data_colors)
        self.data_markers = ['o', 's', 't', 'd', '+', 'p', 'h', 'x', 'star', 't1', 't2', 't3']
        self.marker_choice = len(self.data_markers)
        self.data_references = data_references
        self.reference_choice = len(self.data_references)
        self.data_standard_unit = data_standard_unit
        self.data_buffer_size = data_buffer_size
        self.data_interval = data_interval
        self.update_frequency = update_frequency
        self.plot_together = plot_together
        self.plot_styles = [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DashDotLine, QtCore.Qt.DotLine]
        self.plot_pens = list()
        for idx_data in range(self.data_choice):
            self.plot_pens.append(pg.mkPen(color=self.data_colors[idx_data], width=self.plot_linewidth, style=self.plot_styles[idx_data]))

        # check whether the dimensions of data match
        if self.data_choice != self.reference_choice or self.data_choice != len(self.ylabel):
            raise ValueError

        # construct and set the app, window
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.setWindowTitle(self.win_title)
        self.win.setGeometry(QtCore.QRect(0, 0, self.win_size[0], self.win_size[1]))
        pg.setConfigOption('foreground', self.win_foreground)
        pg.setConfigOption('background', self.win_background)
        pg.setConfigOption('antialias', self.antialias)
        pg.setConfigOption('useOpenGL', self.useOpenGL)

        # add plot and show the plot
        self.data_buffer = []
        self.data_plot = np.zeros((self.data_choice, 1))
        self.plots = []
        self.curves = []
        if self.plot_together:
            plot_instance = self.win.addPlot()
            plot_instance.setLabel('bottom', self.xlabel, self.xlabel_units[0] if self.data_standard_unit else self.xlabel_units[1])
            plot_instance.setLabel('left', ' & '.join(self.ylabel), ' & '.join([self.ylabel_units.get(i)[0] for i in self.ylabel]) if self.data_standard_unit else '&'.join([self.ylabel_units.get(i)[1] for i in self.ylabel]))
            plot_instance.addLegend(offset=(10, 10))
            self.plots.append(plot_instance)
            for idx_data in range(self.data_choice):
                plot_instance.addLine(y=self.data_references[idx_data], pen=self.plot_pens[idx_data])
                curve_instance = self.plots[0].plot(self.data_plot[idx_data], pen=self.plot_pens[idx_data], name=self.data_labels[idx_data])
                # curve_instance = self.plots[0].plot(self.data_plot[idx_data], pen=self.plot_pens[idx_data], symbol=self.data_markers[idx_data], symbolBrush=self.data_colors[idx_data], name=self.data_labels[idx_data])
                self.curves.append(curve_instance)
        else:
            for idx_data in range(self.data_choice):
                plot_instance = self.win.addPlot()
                plot_instance.setLabel('bottom', self.xlabel, self.xlabel_units[0] if self.data_standard_unit else self.xlabel_units[1])
                plot_instance.setLabel('left', self.ylabel[idx_data], self.ylabel_units.get(self.ylabel[idx_data])[0] if self.data_standard_unit else self.ylabel_units.get(self.ylabel[idx_data])[1])
                plot_instance.addLegend(offset=(10, 10))
                plot_instance.addLine(y=self.data_references[idx_data], pen=self.plot_pens[idx_data])
                self.plots.append(plot_instance)
                self.win.nextRow()
                curve_instance = self.plots[idx_data].plot(self.data_plot[idx_data], pen=self.plot_pens[idx_data], name=self.data_labels[idx_data])
                # curve_instance = self.plots[idx_data].plot(self.data_plot[idx_data], pen=self.plot_pens[idx_data], symbol=self.data_markers[idx_data], symbolBrush=self.data_colors[idx_data], name=self.data_labels[idx_data])
                self.curves.append(curve_instance)

        # set the application and update callback
        self.win_app = pg.mkQApp()
        # self.set_update_callback()
        # self.run_app()

    def run_app(self):
        sys.exit(self.win_app.exec_())
        # pg.exec()

    def set_update_callback(self, time, data):
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(lambda: self.update_plot(time, data))
        self.update_timer.start(1 / self.update_frequency)  # data update frequency

    def update_plot(self, timestep, data):
        # check whether the dimensions of data match
        if len(data) != self.data_choice:
            raise ValueError
        # store the data
        self.data_buffer.append(data)
        if len(self.data_buffer) > self.data_buffer_size:
            del self.data_buffer[0]
        # parse the data
        self.data_plot = np.array(self.data_buffer).T
        # update the data
        for idx_data in range(self.data_choice):
            self.curves[idx_data].setData(self.data_plot[idx_data])
            self.curves[idx_data].setPos(timestep, 0)
        # show the plot
        self.win.show()
        QtGui.QApplication.processEvents()

    def close_window(self):
        self.win_app.closeAllWindows()
