#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout
from PyQt5 import uic
import sys
import numpy as np
import pyqtgraph as pg


class MicrInst(QMainWindow):
    def __init__(self):
        super(MicrInst, self).__init__()
        uic.loadUi("wi.ui", self)
        self.show()
        self.plot_area()

        # constants definitions
        self.c = 299792458  # m/s
        self.mc = 0.511e6   # eV/c
        self.Qe = 1.60217662e-19    # elementary charge in Coulombs
        self.p0 = 400e6  # eV/c
        self.L = 27  # m, damping ring perimeter
        self.alpha_p = 1 / 36 - 1 / ((self.p0/self.mc)**2)  # momentum compactor factor
        self.h = 1
        self.eVrf = 9.51e3  # eV, RF voltage
        self.sr_dump = 1.8e3  # eV, SR dumping
        self.phi0 = np.pi / 2

        # Electron beam def
        self.Ne = 2e10  # particles number
        self.N = 3000   # particles number in this simulation

        self.sigma_z = 0.7  # m
        self.sigma_dp = 0.004   # momentum spread

        # initial beam
        self.z0 = np.random.normal(scale=self.sigma_z, size=self.N)
        self.dp0 = np.random.normal(scale=self.sigma_dp, size=self.N)
        self.dp_plot.plot(self.z0, 100*self.dp0, pen=None, symbol='o')

        self.curr_z, self.I = self.get_curr(self.z0)
        self.curr_plot.plot(self.curr_z, self.I, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

        self.data2plot = self.cav_turns(self.spin_n_turns.value())

        # callbacks
        self.spin_n_turns.valueChanged.connect(self.turn_recalc)
        self.spin_turn.valueChanged.connect(self.turn_replot)

    def get_curr(self, z, z_bin=0.03, z_min=-15, z_max=15):
        # all units in meter
        hist, bins = np.histogram(z, range=(z_min, z_max), bins=int((z_max-z_min)/z_bin))
        Qm = self.Qe * self.Ne / self.N
        I = hist * Qm / (z_bin/self.c)

        return bins, I

    def plot_area(self):
        self.plot_window = pg.GraphicsLayoutWidget(parent=self)
        # momentum spread
        self.dp_plot = self.plot_window.addPlot(enableMenu=False)
        self.dp_plot.showGrid(x=True, y=True)
        self.dp_plot.setLabel('left', "dp/p", units='%')
        self.dp_plot.setLabel('bottom', "z", units='m')
        self.dp_plot.setRange(yRange=[-1.5, 1.5])
        self.dp_plot.setRange(xRange=[-12, 12])

        self.plot_window.nextRow()
        # current distribution
        self.curr_plot = self.plot_window.addPlot(enableMenu=False)
        self.curr_plot.showGrid(x=True, y=True)
        self.curr_plot.setLabel('left', "I", units='A')
        self.curr_plot.setLabel('bottom', "z", units='m')
        self.curr_plot.setRange(yRange=[0, 1])
        self.curr_plot.setRange(xRange=[-12, 12])

        p = QHBoxLayout()
        self.output.setLayout(p)
        p.addWidget(self.plot_window)

    def cav_turns(self, n_turns=2000):
        data2plot = {}
        z = self.z0
        dp = self.dp0

        for turn in range(n_turns + 1):
            data2plot[turn] = (z, dp)
            phi = self.phi0 - 2*np.pi*self.h*z/self.L
            dp = dp + self.eVrf*np.cos(phi)/self.p0 - self.sr_dump/self.p0
            z = z - self.L*self.alpha_p*dp
            self.status_bar.showMessage("turn = %g %%" % (100*turn/n_turns))
        return data2plot

    def turn_recalc(self):
        self.spin_turn.setMaximum(self.spin_n_turns.value())
        self.slider_turn.setMaximum(self.spin_n_turns.value())
        self.data2plot = self.cav_turns(self.spin_n_turns.value())

    def turn_replot(self):
        z, dp = self.data2plot[self.spin_turn.value()]
        self.dp_plot.clear()
        self.dp_plot.plot(z, 100 * dp, pen=None, symbol='o')
        self.curr_plot.clear()
        curr_z, I = self.get_curr(z)
        self.curr_plot.plot(curr_z, I, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))


if __name__ == "__main__":
    app = QApplication(['mv'])
    w = MicrInst()
    sys.exit(app.exec_())
