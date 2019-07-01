#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout
from PyQt5 import uic
import sys
import numpy as np
import pyqtgraph as pg
# import pyximport; pyximport.install()
# import cav_turn


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
        # self.alpha_p = 1 / 36 - 1 / ((self.p0/self.mc)**2)  # momentum compactor factor
        self.h = 1
        self.eVrf = 9.51e3  # eV, RF voltage
        self.sr_dump = 1.8e3  # eV, SR dumping
        self.phi0 = np.pi / 2
        self.dz = 0.03

        # wake properties
        self.wake = []
        self.v = []
        self.L_wake = 10  # m
        self.Fr = self.spin_freq.value() * 1e9
        self.Rsh = self.spin_Rsh.value() * 1e3
        self.alpha_p = self.spin_alpha_p.value()
        self.Q = self.spin_Q.value()

        # Electron beam def
        self.custom_dist = 'default'
        self.Ne = 2e10  # particles number
        self.N = 5000   # particles number in this simulation

        self.sigma_z = 0.6  # m
        self.sigma_dp = 0.004   # momentum spread
        # initial beam
        self.beam_particles_dist()

        self.wake2plot, self.curr2plot, self.dp2plot = self.cav_turns(self.spin_n_turns.value())

        self.dp_plot.plot(self.z0, 100 * self.dp0, pen=None, symbol='star', symbolSize=5)
        self.curr_plot.plot(self.curr_z, self.I, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        self.wake_plot.plot(self.xi, self.wake / 1e12, pen=pg.mkPen('g', width=1))
        self.wake_curr_plot.plot(self.zv, self.v / 1e3, pen=pg.mkPen('r', width=1))

        # callbacks
        self.button_calculate.clicked.connect(self.turn_recalc)
        self.spin_turn.valueChanged.connect(self.turn_replot)
        self.rb_linac_beam.toggled.connect(self.beam_type)
        
    def beam_type(self):
        if self.rb_linac_beam.isChecked():
            self.custom_dist = 'linac_beam'
        else:
            self.custom_dist = 'default'
        self.beam_particles_dist()

    def beam_particles_dist(self):
        if self.custom_dist == 'default':
            self.dz = 0.03
            self.sigma_z = 1
            self.z0 = np.random.normal(scale=self.sigma_z, size=self.N)
        elif self.custom_dist == 'linac_beam':
            n_local = int(self.N / 15)  # I wanna see 15 bunches
            self.N = n_local * 15
            self.dz = 0.006
            self.sigma_z = 0.0006
            curr_z0 = np.array([])
            for i in range(-7, 8):
                curr_z0 = np.append(curr_z0, np.random.normal(loc=0.105*i, scale=self.sigma_z, size=n_local))
            self.z0 = curr_z0
        else:
            print('u should not be here')
        self.dp0 = np.random.normal(scale=self.sigma_dp, size=self.N)

        # init beam
        self.curr_z, self.I = self.get_curr(self.z0)

        # init wake
        self.xi = np.linspace(-1 * self.L_wake, 0, int(self.L_wake / self.dz))
        self.wake = self.calc_wake(self.xi)
        # init wake convolution

        self.v = - np.convolve(self.wake, self.I) * self.dz / self.c
        self.zv = np.linspace(max(self.curr_z) - self.dz * len(self.v), max(self.curr_z), len(self.v))

    def get_curr(self, z, z_min=-15, z_max=15):
        # all units in meter
        hist, bins = np.histogram(z, range=(z_min, z_max), bins=int((z_max-z_min)/self.dz))
        Qm = self.Qe * self.Ne / self.N
        I = hist * Qm / (self.dz/self.c)

        return bins, I

    def plot_area(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_window = pg.GraphicsLayoutWidget(parent=self)
        # wake
        self.wake_plot = self.plot_window.addPlot(enableMenu=False)
        self.wake_plot.showGrid(x=True, y=True)
        self.wake_plot.setLabel('left', "W", units='V/pC')
        self.wake_plot.setLabel('bottom', "z", units='m')
        self.wake_plot.setRange(yRange=[-20, 20])
        self.wake_plot.setRange(xRange=[-10, 0])

        self.plot_window.nextRow()
        # wake_curr convolve
        self.wake_curr_plot = self.plot_window.addPlot(enableMenu=False)
        self.wake_curr_plot.showGrid(x=True, y=True)
        self.wake_curr_plot.setLabel('left', "V", units='kV')
        self.wake_curr_plot.setLabel('bottom', "z", units='m')
        self.wake_curr_plot.setRange(yRange=[-10, 10])
        self.wake_curr_plot.setRange(xRange=[-8, 8])

        self.plot_window.nextRow()
        # current distribution
        self.curr_plot = self.plot_window.addPlot(enableMenu=False)
        self.curr_plot.showGrid(x=True, y=True)
        self.curr_plot.setLabel('left', "I", units='A')
        self.curr_plot.setLabel('bottom', "z", units='m')
        self.curr_plot.setRange(yRange=[0, 1])
        self.curr_plot.setRange(xRange=[-3, 3])

        self.plot_window.nextRow()
        # momentum spread
        self.dp_plot = self.plot_window.addPlot(enableMenu=False)
        self.dp_plot.showGrid(x=True, y=True)
        self.dp_plot.setLabel('left', "dp/p", units='%')
        self.dp_plot.setLabel('bottom', "z", units='m')
        self.dp_plot.setRange(yRange=[-1.5, 1.5])
        self.dp_plot.setRange(xRange=[-8, 8])

        p = QHBoxLayout()
        self.output.setLayout(p)
        p.addWidget(self.plot_window)

    def cav_turns(self, n_turns=5000):
        dp2plot = {}
        curr2plot = {}
        wake2plot = {}
        z = self.z0
        dp = self.dp0

        for turn in range(n_turns + 1):
            dp2plot[turn] = (z, dp)
            phi = self.phi0 - 2*np.pi*self.h*z/self.L
            # cavity
            dp = dp + self.eVrf*np.cos(phi)/self.p0 - self.sr_dump/self.p0
            # wakefield
            curr_z, I = self.get_curr(z)
            curr2plot[turn] = (curr_z, I)
            v = - np.convolve(self.wake, I) * self.dz / self.c
            wake2plot[turn] = (self.zv, v)
            v_s = np.interp(z, self.zv, v)
            dp = dp + v_s / self.p0

            z = z - self.L*self.alpha_p*dp
            self.status_bar.showMessage("turn = %g %%" % (100*turn/n_turns))
        return wake2plot, curr2plot, dp2plot

    def turn_recalc(self):
        self.spin_turn.setMaximum(self.spin_n_turns.value())
        self.slider_turn.setMaximum(self.spin_n_turns.value())

        self.Fr = self.spin_freq.value() * 1e9
        self.Rsh = self.spin_Rsh.value() * 1e3
        self.alpha_p = self.spin_alpha_p.value()
        self.Q = self.spin_Q.value()

        # wake
        self.wake = self.calc_wake(self.xi)
        self.wake_plot.clear()
        self.wake_plot.plot(self.xi, self.wake / 1e12, pen=pg.mkPen('g', width=1))

        # wake_curr convolution
        self.v = - np.convolve(self.wake, self.I) * self.dz / self.c

        self.wake2plot, self.curr2plot,  self.dp2plot = self.cav_turns(self.spin_n_turns.value())
        self.turn_replot()

    def turn_replot(self):
        z, dp = self.dp2plot[self.spin_turn.value()]
        self.dp_plot.clear()
        self.dp_plot.plot(z, 100 * dp, pen=None, symbol='star', symbolSize=5)

        curr_z, I = self.curr2plot[self.spin_turn.value()]
        self.curr_plot.clear()
        self.curr_plot.plot(curr_z, I, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        # self.curr_plot.plot(z)

        zv, v = self.wake2plot[self.spin_turn.value()]
        self.wake_curr_plot.clear()
        self.wake_curr_plot.plot(zv, v / 1e3, pen=pg.mkPen('r', width=1))

    def calc_wake(self, xi):
        wr = 2 * np.pi * self.Fr
        alpha = wr / (2 * self.Q)
        wr_1 = wr * np.sqrt(1 - 1 / (4 * self.Q ** 2))

        wake = 2 * alpha * self.Rsh * np.exp(alpha * xi / self.c) * (np.cos(wr_1 * xi / self.c) + (alpha / wr_1) *
                                                                     np.sin(wr_1 * xi / self.c))
        wake[xi == 0] = alpha * self.Rsh
        wake[xi > 0] = 0

        return wake


if __name__ == "__main__":
    app = QApplication(['mv'])
    w = MicrInst()
    sys.exit(app.exec_())
