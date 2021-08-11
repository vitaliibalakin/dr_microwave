#!/usr/bin/env python3

from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout
from PyQt5 import uic, QtGui
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
        self.p0 = 392e6  # eV/c
        self.L = 27  # m, damping ring perimeter
        # self.alpha_p = 1 / 36 - 1 / ((self.p0/self.mc)**2)  # momentum compactor factor
        self.h = 1
        self.eVrf = 9.51e3  # eV, RF voltage
        self.sr_dump = 1.878e3  # eV, SR dumping
        self.phi0 = np.arccos(self.sr_dump/self.eVrf)
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
        self.Ne = 1e10  # particles number
        self.N = 20000   # particles number in this simulation

        self.sigma_z = 0.3  # m
        self.sigma_dp = 3.5e-4   # momentum spread
        # initial beam
        self.beam_particles_dist()

        # measured beam profiles
        # mes_data = np.loadtxt('exclusive_profs.txt')
        mes_data = [0, 0, 0, 0, 0]
        i = 0
        f = open('exclusive_profs.txt', 'r')
        for line in f.readlines():
            mes_data[i] = np.asarray(line.split(' '), dtype=np.float64)
            i += 1
        # print(mes_data)
        self.mes_profs = {100: mes_data[0], 500: mes_data[1], 800: mes_data[2], 1600: mes_data[3], 3200: mes_data[4]}
        # self.dp_plot.plot(mes_data[4])
        # self.curr_plot.plot(mes_data[1])
        # self.wake_plot.plot(mes_data[2])
        # self.wake_curr_plot.plot(mes_data[3])

        # self.optimize()

        # sim optimization is under construction
        # self.wake2plot, self.curr2plot, self.dp2plot = self.cav_turns(self.spin_n_turns.value())
        #
        # self.dp_plot.plot(self.z0, 100 * self.dp0, pen=None, symbol='star', symbolSize=5)
        # self.curr_plot.plot(self.curr_z, self.I, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        # self.wake_plot.plot(self.xi, self.wake / 1e12, pen=pg.mkPen('g', width=3))
        # self.wake_curr_plot.plot(self.zv, self.v / 1e3, pen=pg.mkPen('r', width=3))

        self.wake2plot = {}
        self.curr2plot = {}
        self.dp2plot = {}

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
            self.sigma_z = 0.2
            self.z0 = np.random.normal(scale=self.sigma_z, size=self.N)
            self.dp0 = np.random.normal(scale=self.sigma_dp, size=self.N)
        elif self.custom_dist == 'linac_beam':
            # n_local = int(self.N / 15)  # I wanna see 15 bunches
            # self.N = n_local * 15
            self.dz = 0.005
            self.sigma_z = 3e-3
            curr_z0 = np.array([])
            self.modul = np.random.normal(loc=0, scale=0.46, size=self.N)
            self.hist, self.bins = np.histogram(self.modul, bins=15)
            # self.hist = [1, 10, 17, 93, 303, 597, 1006, 1233, 1179, 829, 475, 187, 56, 11, 3]
            # print(self.hist)
            for i in range(-7, 8):
                curr_z0 = np.append(curr_z0, np.random.normal(loc=0.105*i, scale=self.sigma_z, size=self.hist[i+7]))
            print(len(curr_z0))
            self.z0 = curr_z0
            # self.z0 = np.loadtxt('model_linac.txt')
            # self.dp0 = np.loadtxt('model_linac_dp.txt')
            # np.savetxt('model_linac.txt', curr_z0)
        else:
            print('u should not be here')
        self.dp0 = np.random.normal(scale=self.sigma_dp, size=self.N)
        print(self.dp0)
        # np.savetxt('model_linac_dp.txt', self.dp0)

        # init beam
        self.curr_z, self.I = self.get_curr(self.z0)
        # init wake
        self.xi = np.linspace(-1 * self.L_wake, 0, int(self.L_wake / self.dz))

    def get_curr(self, z, z_min=-10, z_max=10):
        # all units in meter
        hist, bins = np.histogram(z, range=(z_min, z_max), bins=int((z_max-z_min)/self.dz))
        Qm = self.Qe * self.Ne / self.N
        I = hist * Qm / (self.dz/self.c)

        return bins, I

    def plot_area(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)
        label_style = {'font-size': '16pt'}
        font = QtGui.QFont()
        font.setPixelSize(16)

        self.plot_window = pg.GraphicsLayoutWidget(parent=self)
        # wake
        self.wake_plot = self.plot_window.addPlot(enableMenu=False)
        self.wake_plot.showGrid(x=True, y=True)
        self.wake_plot.setLabel('left', "W", units='V/pC', **label_style)
        self.wake_plot.setLabel('bottom', "z", units='m', **label_style)
        self.wake_plot.setRange(yRange=[-20, 20])
        self.wake_plot.setRange(xRange=[-10, 0])
        self.wake_plot.getAxis("bottom").tickFont = font
        self.wake_plot.getAxis("left").tickFont = font

        self.plot_window.nextRow()
        # wake_curr convolve
        self.wake_curr_plot = self.plot_window.addPlot(enableMenu=False)
        self.wake_curr_plot.showGrid(x=True, y=True)
        self.wake_curr_plot.setLabel('left', "V", units='kV', **label_style)
        self.wake_curr_plot.setLabel('bottom', "z", units='m', **label_style)
        self.wake_curr_plot.setRange(yRange=[-12, 18])
        self.wake_curr_plot.setRange(xRange=[-6, 4])
        self.wake_curr_plot.getAxis("bottom").tickFont = font
        self.wake_curr_plot.getAxis("left").tickFont = font

        self.plot_window.nextRow()
        # current distribution
        self.curr_plot = self.plot_window.addPlot(enableMenu=False)
        self.curr_plot.showGrid(x=True, y=True)
        self.curr_plot.setLabel('left', "I", units='A', **label_style)
        self.curr_plot.setLabel('bottom', "z", units='m', **label_style)
        self.curr_plot.setRange(yRange=[0, 2])
        self.curr_plot.setRange(xRange=[-4, 4])
        self.curr_plot.getAxis("bottom").tickFont = font
        self.curr_plot.getAxis("left").tickFont = font

        self.plot_window.nextRow()
        # momentum spread
        self.dp_plot = self.plot_window.addPlot(enableMenu=False)
        self.dp_plot.showGrid(x=True, y=True)
        self.dp_plot.setLabel('left', "dp/p (%)", **label_style)
        self.dp_plot.setLabel('bottom', "z", units='m', **label_style)
        self.dp_plot.setRange(yRange=[-1.5, 1.5])
        self.dp_plot.setRange(xRange=[-6, 6])
        self.dp_plot.getAxis("bottom").tickFont = font
        self.dp_plot.getAxis("left").tickFont = font

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
            dp = dp + self.eVrf*np.cos(phi)/self.p0 - self.sr_dump*(1 + dp)**4/self.p0
            # wakefield
            curr_z, I = self.get_curr(z)
            curr2plot[turn] = (curr_z, I)
            v = - np.convolve(self.wake, I) * self.dz / self.c
            zv = np.linspace(max(curr_z) - self.dz * len(v), max(curr_z), len(v))
            wake2plot[turn] = (zv, v)
            v_s = np.interp(z, zv, v)
            dp = dp + v_s / self.p0
            z = z - self.L*self.alpha_p*dp

            print("turn = %g %%" % (100*turn/n_turns))
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

        # # wake_curr convolution
        # self.v = - np.convolve(self.wake, self.I) * self.dz / self.c
        self.wake2plot.clear()
        self.curr2plot.clear()
        self.dp2plot.clear()

        self.wake2plot, self.curr2plot,  self.dp2plot = self.cav_turns(self.spin_n_turns.value())
        self.turn_replot()

    def turn_replot(self):
        z, dp = self.dp2plot[self.spin_turn.value()]
        # self.dp_plot.clear()
        # self.dp_plot.plot(z, 100 * dp, pen=None, symbol='star', symbolSize=5)

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
        print(alpha)
        wake = 2 * alpha * self.Rsh * np.exp(alpha * xi / self.c) * (np.cos(wr_1 * xi / self.c) + (alpha / wr_1) *
                                                                     np.sin(wr_1 * xi / self.c))
        wake[xi == 0] = alpha * self.Rsh
        wake[xi > 0] = 0

        return wake

    def optimize(self, f_init=2.5e9, q_init=5.2, r_sh_init=3.9e4):
        # 1st opt for freq, then for q, then for r_sh
        # beam begins with val = 0.04 from curr
        gamma = 1e11
        self.Fr = f_init
        self.Q = q_init
        self.Rsh = r_sh_init

        q_step = 0.1
        r_sh_step = 0.1e3

        def aim_func(I, mes_profs):
            s = 0
            for k, v in mes_profs.items():
                data = I[k][1]
                fp = data[np.where(data > 0.04)[0][0]:np.where(data > 0.04)[0][-1]]
                xp = 0.005 * np.arange(len(fp))
                x = np.linspace(0, max(xp), len(v), endpoint=True)
                shared = np.interp(x, xp, fp)
                s = s + np.sum((v - shared) ** 2)
            # self.curr_plot.clear()
            # self.wake_plot.clear()
            # self.curr_plot.plot(x, shared[:-1], stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
            # self.wake_plot.plot(xp, fp[:-1], stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
            return s

        # initial step
        # new culc run for 1601 turn
        # self.wake = self.calc_wake(self.xi)
        # wake, curr, dp = self.cav_turns(3201)
        # criteria = aim_func(curr, self.mes_profs)

        # self.Fr += 0.1e9
        # self.wake = self.calc_wake(self.xi)
        # wake, curr, dp = self.cav_turns(3201)
        # crit_new = aim_func(curr, self.mes_profs)

        # grad = (crit_new - criteria) / 0.1e9

        # gradient descent
        # step = grad * gamma
        # self.Fr -= step
        # criteria = crit_new
        x_arr = np.zeros([100, ])
        Psy = np.zeros([100, ])
        for i in range(1, 101):
            self.Fr = 1e8 * i
            # self.Rsh = 1e3 * i
            # self.Q = 1 + 0.1 * i
            self.wake = self.calc_wake(self.xi)
            wake, curr, dp = self.cav_turns(3201)
            crit_new = aim_func(curr, self.mes_profs)
            print(self.Fr/1e9, crit_new)
            # print(self.Rsh, crit_new)
            # print(self.Q, crit_new)
            x_arr[i-1] = self.Fr/1e9
            # x_arr[i-1] = self.Rsh
            # x_arr[i-1] = self.Q
            Psy[i-1] = crit_new
        # for i in range(100):
        #     self.wake = self.calc_wake(self.xi)
        #     wake, curr, dp = self.cav_turns(1601)
        #     crit_new = aim_func(curr, self.mes_profs)
        #     grad1 = (crit_new - criteria) / step
        #
        #     gamma = abs(step / (grad1 - grad))
        #     grad = grad1
        #     step = grad * gamma
        #     self.Fr -= step
        #     criteria = crit_new
        #     if grad == 0:
        #         break
        #     print(self.Fr/1e9, criteria, step/1e9, grad, gamma/1e11)
        np.savetxt('func_vs_freq.txt', np.vstack((x_arr, Psy)))
        sys.exit()


if __name__ == "__main__":
    app = QApplication(['mv'])
    w = MicrInst()
    sys.exit(app.exec_())
