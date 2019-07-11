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

        mes_data = [0, 0, 0, 0, 0]
        i = 0
        f = open('exclusive_profs.txt', 'r')
        for line in f.readlines():
            mes_data[i] = np.asarray(line.split(' '), dtype=np.float64)
            i += 1
        # print(mes_data)
        self.mes_profs = {100: mes_data[0], 500: mes_data[1], 800: mes_data[2], 1600: mes_data[3], 3200: mes_data[4]}

        self.plot_win = pg.GraphicsWindow()
        self.I_plot = self.plot_win.addPlot()
        self.I_plot.showGrid(x=True, y=True)
        self.I_plot.setLabel('bottom', "Length", units='m')
        self.I_plot.setLabel('left', "Current", units='a.u.')
        self.I_plot.addLegend()

        x = np.arange(0, len(mes_data[4]), 1) * 0.025268554788203976 / 3
        self.I_plot.plot(x, mes_data[4], pen=pg.mkPen('b', width=2))


if __name__ == "__main__":
    app = QApplication(['kek'])
    w = Plot()
    sys.exit(app.exec_())
