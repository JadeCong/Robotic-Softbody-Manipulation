# import pyformulas as pf
# import matplotlib.pyplot as plt
# import numpy as np
# import time
#
#
# fig = plt.figure()
# canvas = np.zeros((480, 640))
# screen = pf.screen(canvas, 'Contact Force')
#
# start = time.time()
# while True:
#     now = time.time() - start
#
#     x = np.linspace(now-2, now, 100)
#     y = np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
#     plt.xlim(now-2,now+1)
#     plt.ylim(-3,3)
#     plt.plot(x, y, c='black')
#
#     # If we haven't already shown or saved the plot, then we need to draw the figure first...
#     fig.canvas.draw()
#
#     image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#
#     screen.update(image)

#screen.close()



# from utils.rtplot import RealtimePlotData
# import time as t
# import numpy as np
#
#
# rtpd = RealtimePlotData(title="Ultrasound Scanning Sim Data")
# start = t.time()
# old_time = 0
# old_data1 = 0
# old_data2 = 0
# old_data3 = 0
#
# while True:
#     # y = np.sin(2 * np.pi * x) + np.sin(3 * np.pi * x)
#     # y1 = np.sin(3 * np.pi * x) + np.sin(4 * np.pi * x)
#     now_time = t.time() - start
#     # time = np.array([old_time, now_time])
#     time = np.linspace(old_time, now_time, 2)
#
#     now_data1 = np.sin(2 * np.pi * now_time) + np.sin(3 * np.pi * now_time)
#     now_data2 = np.sin(1 * np.pi * now_time) + 1.5 * np.cos(4 * np.pi * now_time)
#     now_data3 = 2* np.sin(1.5 * np.pi * now_time) + np.cos(5 * np.pi * now_time)
#     # data = np.array([old_data, now_data])
#     data1 = np.linspace(old_data1, now_data1, 2)
#     data2 = np.linspace(old_data2, now_data2, 2)
#     data3 = np.linspace(old_data3, now_data3, 2)
#     data = np.array([data1, data2, data3])
#
#     rtpd.plot_data(time, data)
#
#     old_time = now_time
#     old_data1 = now_data1
#     old_data2 = now_data2
#     old_data3 = now_data3
#
# rtpd.end_plot()



# from utils.dynplot import dynplot
# from math import sin, pi
#
# dplt = dynplot()
# for i in range(1000):
#     x = range(i, i+20)
#     y1 = [sin(2*pi*x/20) for x in x]
#     y2 = [sin(2 * pi * x / 10) for x in x]
#     dplt.plot(y1)
#     dplt.plot(y2)
#     _ = dplt.ax.set_title('Wave')
#     dplt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import threading
# import sys
# from random import random, randrange
# from time import sleep
#
# '''
# 绘制2x2的画板
# 可设置窗口标题和4个子图标题
# 可更新曲线数据
# '''
# quit_flag = False  # 退出标志
#
#
# class Plot2_2(object):
#     """ 2x2的画板 """
#
#     def __init__(self, wtitle='Figure', p1title='1', p2title='2', p3title='3',
#                  p4title='4'):
#         self.sub_title = [p1title, p2title, p3title, p4title]  # 4个子图的标题
#         self.fig, self.ax = plt.subplots(2, 2)  # 创建2X2子图
#         self.fig.subplots_adjust(wspace=0.3, hspace=0.3)  # 设置子图之间的间距
#         self.fig.canvas.set_window_title(wtitle)  # 设置窗口标题
#
#         # 子图字典，key为子图的序号，value为子图句柄
#         self.axdict = {0: self.ax[0, 0], 1: self.ax[0, 1], 2: self.ax[1, 0], 3: self.ax[1, 1]}
#
#     def showPlot(self):
#         """ 显示曲线 """
#         plt.show()
#
#     def setPlotStyle(self, index):
#         """ 设置子图的样式，这里仅设置了标题 """
#         self.axdict[index].set_title(self.sub_title[index], fontsize=12)
#
#     def updatePlot(self, index, x, y):
#         """
#         更新指定序号的子图
#         :param index: 子图序号
#         :param x: 横轴数据
#         :param y: 纵轴数据
#         :return:
#         """
#         # X轴数据必须和Y轴数据长度一致
#         if len(x) != len(y):
#             ex = ValueError("x and y must have same first dimension")
#             raise ex
#
#         self.axdict[index].cla()  # 清空子图数据
#         self.axdict[index].plot(x, y)  # 绘制最新的数据
#         self.setPlotStyle(index)  # 设置子图样式
#         if min(x) < max(x):
#             self.axdict[index].set_xlim(min(x), max(x))  # 根据X轴数据区间调整X轴范围
#         plt.draw()
#         print("%s end" % sys._getframe().f_code.co_name)
#
#
# def updatePlot(plot):
#     """
#     模拟收到实时数据，更新曲线的操作
#     :param plot: 曲线实例
#     :return:
#     """
#     print("Thread: %s" % threading.current_thread().getName())
#     count = 0
#     global quit_flag
#     print("quit_flag[%s]" % str(quit_flag))
#     while True:
#         if quit_flag:
#             print("quit_flag[%s]" % str(quit_flag))
#             break
#         count += 1
#         print("count#%d" % count)
#         x = np.arange(0, 100, 1)
#         y = np.random.normal(loc=1, scale=1, size=100)  # 产生随机数，模拟变化的曲线
#         index = randrange(4)  # 随机更新某一个子图
#         plot.updatePlot(index, x, y)
#         sleep(random() * 3)
#
#
# def main():
#     p = Plot2_2()  # 创建一个2X2画板
#
#     t = threading.Thread(target=updatePlot, args=(p,))  # 启动一个线程更新曲线数据
#     t.start()
#
#     p.showPlot()  # showPlot方法会阻塞当前线程，直到窗口关闭
#     print("plot close")
#     global quit_flag
#     quit_flag = True  # 通知更新曲线数据的线程退出
#
#     t.join()
#     print("Thread: %s end" % threading.current_thread().getName())
#
#
# if __name__ == '__main__':
#     main()



# from PyQt5.Qt import *
# from pyqtgraph import PlotWidget
# from PyQt5 import QtCore
# import numpy as np
# import pyqtgraph as pq
#
#
# class Window(QWidget):
#     def __init__(self):
#         super().__init__()
#         # 设置下尺寸
#         self.resize(600,600)
#         # 添加 PlotWidget 控件
#         self.plotWidget_ted = PlotWidget(self)
#         # 设置该控件尺寸和相对位置
#         self.plotWidget_ted.setGeometry(QtCore.QRect(25,25,550,550))
#
#         # 仿写 mode1 代码中的数据
#         # 生成 300 个正态分布的随机数
#         self.data1 = np.random.normal(size=300)
#
#         self.curve2 = self.plotWidget_ted.plot(self.data1, labels="Data", name="mode2", background="white")
#         self.ptr1 = 0
#
#         # 设定定时器
#         self.timer = pq.QtCore.QTimer()
#         # 定时器信号绑定 update_data 函数
#         self.timer.timeout.connect(self.update_data)
#         # 定时器间隔50ms，可以理解为 50ms 刷新一次数据
#         self.timer.start(50)
#
#     # 数据左移
#     def update_data(self):
#         self.data1[:-1] = self.data1[1:]
#         self.data1[-1] = np.random.normal()
#         # 数据填充到绘制曲线中
#         self.curve2.setData(self.data1)
#         # x 轴记录点
#         self.ptr1 += 1
#         # 重新设定 x 相关的坐标原点
#         self.curve2.setPos(self.ptr1,0)
#
#
# if __name__ == '__main__':
#     import sys
#     # PyQt5 程序固定写法
#     app = QApplication(sys.argv)
#
#     # 将绑定了绘图控件的窗口实例化并展示
#     window = Window()
#     window.show()
#
#     # PyQt5 程序固定写法
#     sys.exit(app.exec())



# import pyqtgraph as pg
# from PyQt5.Qt import *
# from pyqtgraph.Qt import QtCore, QtGui
# import numpy as np
#
# win = pg.GraphicsLayoutWidget(show=True)
# win.setWindowTitle('pyqtgraph example: Scrolling Plots')
# win.setBackground('w')
#
# # 1) Simplest approach -- update data in the array such that plot appears to scroll
# #    In these examples, the array size is fixed.
# p1 = win.addPlot()
# p2 = win.addPlot()
# data1 = np.random.normal(size=300)
# curve1 = p1.plot(pen='g')
# curve2 = p2.plot(pen='r')
# ptr1 = 0
#
#
# def update1():
#     global data1, ptr1
#     data1[:-1] = data1[1:]  # shift data in the array one sample left
#     # (see also: np.roll)
#     data1[-1] = np.random.normal()
#     curve1.setData(data1)
#
#     ptr1 += 1
#     curve2.setData(data1)
#     curve2.setPos(ptr1, 0)
#
#
# # 3) Plot in chunks, adding one new plot curve for every 100 samples
# chunkSize = 100
# # Remove chunks after we have 10
# maxChunks = 10
# startTime = pg.ptime.time()
# win.nextRow()
# p5 = win.addPlot(colspan=2)
# p5.setLabel('bottom', "Timestep", 's')
# p5.setLabel('left', 'Value', 'N & m/s')
# p5.setXRange(-10, 0)
# curves = []
# data5 = np.empty((chunkSize + 1, 2))
# ptr5 = 0
#
#
# def update3():
#     global p5, data5, ptr5, curves
#     now = pg.ptime.time()
#     for c in curves:
#         c.setPos(-(now - startTime), 0)
#
#     i = ptr5 % chunkSize
#     if i == 0:
#         curve = p5.plot()
#         curves.append(curve)
#         last = data5[-1]
#         data5 = np.empty((chunkSize + 1, 2))
#         data5[0] = last
#         while len(curves) > maxChunks:
#             c = curves.pop(0)
#             p5.removeItem(c)
#     else:
#         curve = curves[-1]
#
#     data5[i + 1, 0] = now - startTime
#     data5[i + 1, 1] = np.random.normal()
#     curve.setData(x=data5[:i + 2, 0], y=data5[:i + 2, 1])
#     ptr5 += 1
#
#
# # update all plots
# def update():
#     update1()
#     update3()
#
#
# timer = pg.QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)

# if __name__ == '__main__':
#     pg.exec()

# if __name__ == '__main__':
#     import sys
#     # PyQt5 程序固定写法
#     app = QApplication(sys.argv)
#
#     # 将绑定了绘图控件的窗口实例化并展示
#     # window = Window()
#     # window.show()
#
#     # PyQt5 程序固定写法
#     sys.exit(app.exec())

# if __name__ == '__main__':
#     import sys
#
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()



import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from utils.qtplot import RealtimePlotWindow
import time
from threading import Thread


qtp = RealtimePlotWindow(ylabel=["Position", "Velocity", "Force"], data_labels=["pos", "vel", "force"], data_references=[0.04, 2, 1], plot_together=False)
timestep = 0
data = np.array([np.random.normal(), np.random.normal(), np.random.normal()])

def sample_data():
    global timestep, data
    timestep += 1
    data = np.array([np.random.normal(), np.random.normal(), np.random.normal()])

# timer = QtCore.QTimer()
# timer.timeout.connect(lambda: qtp.update_plot(timestep, data))
# timer.timeout.connect(sample_data)
# timer.start(50)
#
# qtp.run_window()




# def update():
#     while True:
#         sample_data()
#         time.sleep(0.05)
#
# # main thread
# main_thread = Thread(target=update)
# main_thread.start()
#
# # update theread
# update_thread = Thread(target=qtp.update_plot, args=(timestep, data))
# update_thread.start()
#
# print("fuck")

while True:
    sample_data()
    qtp.update_plot(timestep, data)
    time.sleep(0.05)
