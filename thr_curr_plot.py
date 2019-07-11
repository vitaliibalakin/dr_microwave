#!/usr/bin/env python3

import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys


class Plot:
    def __init__(self):
        super(Plot, self).__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        self.voltage = np.linspace(2, 9.41, 20)
        self.I_th = 0.1722 * self.voltage
        self.I = 0.377 * np.sqrt(self.voltage)

        self.plot_win = pg.GraphicsWindow()
        self.I_plot = self.plot_win.addPlot()
        self.I_plot.showGrid(x=True, y=True)
        self.I_plot.setLabel('bottom', "Voltage", units='kV')
        self.I_plot.setLabel('left', "Peak current", units='A')
        self.I_plot.addLegend()

        self.I_plot.plot(self.voltage, self.I_th, pen=pg.mkPen('r', width=3), name='peak current threshold')
        self.I_plot.plot(self.voltage, self.I, pen=pg.mkPen('b', width=3), name='peak beam current')


if __name__ == "__main__":
    app = QApplication(['kek'])
    w = Plot()
    sys.exit(app.exec_())
