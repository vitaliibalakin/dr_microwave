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

        data = np.loadtxt('func_vs_q.txt')

        self.plot_win = pg.GraphicsWindow()
        self.func_plot = self.plot_win.addPlot()
        self.func_plot.showGrid(x=True, y=True)
        self.func_plot.setLabel('bottom', "Q")#, units='GHz')
        self.func_plot.setLabel('left', "\u03C6", units='a.u.')
        self.func_plot.addLegend(offset=(100, 10))

        self.func_plot.plot(data[0], data[1], pen=pg.mkPen('b', width=2), name='Rs = 57 kOhm, F = 0.42 GHz')


if __name__ == "__main__":
    app = QApplication(['kek'])
    w = Plot()
    sys.exit(app.exec_())
