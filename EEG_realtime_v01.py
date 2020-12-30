"""
EEG "Realtime" Visualizer
-------------------------

@author: Kevin Machado Gamboa
Created on Tue Oct 20 22:31:16 2020
References:

"""
import threading
import mne
from time import time
import sys
import numpy as np
import pyqtgraph as pg
from scipy import signal
# GUI libraries
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
# Own Lobrary
import ppfunctions_1 as ppf

from pyOpenBCI import OpenBCICyton

from os.path import dirname as up

mf_path = (up(__file__)).replace('\\', '/')  # Main Folder Path
# Samples recommended for storing and plotting data 1250 --> 5seg
DT = 5  # Display Window lenght in time
UPT = 0.1  # Update time miliseconds
FS = 250
uVolts_per_count = (4500000) / 24 / (2 ** 23 - 1)  # uV/count
Data = [[0, 0, 0, 0, 0, 0, 0, 0]]

# Notch Filter
def notch_filter(val, data, fs=250):
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        fin = data = signal.lfilter(b, a, data)
    return fin

def bandpass(start, stop, data, fs=250):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.lfilter(b, a, data, axis=0)

def butter_bp_coe(lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter coefficients b and a
    Ref:
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bp_fil(data, lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter
    Ref:
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    b, a = butter_bp_coe(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

class mainWindow(QMainWindow):
    def __init__(self):
        # Inicia el objeto QMainWindow
        QMainWindow.__init__(self)
        # Loads an .ui file & configure UI
        loadUi("EEG_realtime_ui_v0.ui", self)
        #
        #        _translate = QtCore.QCoreApplication.translate
        #        QMainWindow.setWindowTitle('AID')
        # Variables
        self.fs = None
        self.data = None
        self.data_P1 = None
        self.t = 0
        self.d_signal = np.zeros(FS * DT)  # Vector Label
        self.Time = np.zeros(FS * DT)
        self.duration = None
        self.vectortime = None
        self.file_path = str
        self.plot_colors = ['#ffee3d', '#0072bd', '#d95319', '#bd0000']

        # Initializing Thread_Based parallel
        self.x = threading.Thread(target=start_board)
        self.x.daemon = True

        self._configure_plot()
        self.buttons()
        self.timer_1()

        self.openF()

    def _update_plot(self):
        """
        Updates and redraws the graphics in the plot.
        """
        global Data

        t_data = np.array(Data[-int(FS*DT):]).T

        # self.d_signal = np.roll(self.d_signal, 1, 0)
        # self.d_signal[1] = Data
        #
        # self.Time = np.roll(self.Time, 1, 0)
        # self.Time[1] = self.t % (
        #         FS * DT)  # np.linspace((self.t % 100 * (1 / FS)), ((1 + self.t % 100) * (1 / FS)), 1)

        # print("Data: %f, Time: %i " % (self.d_signal[-1:], self.Time[-1:]))

        self._plt1.clear()
        self._plt1.plot(t_data[0], pen=self.plot_colors[0])

        self._plt2.clear()
        self._plt2.plot(butter_bp_fil(notch_filter(50, t_data[0]),5, 25, FS,order=3)[200:], pen=self.plot_colors[0])

        self.t += 1

    # ------------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------------
    def buttons(self):
        """
        Configures the connections between signals and UI elements.
        """
        self.openButton.clicked.connect(self.openF)
        self.playButton.clicked.connect(self.playB)
        self.stopButton.clicked.connect(self.stopB)

    def openF(self):
        """
        open a box to browse the audio file. Then converts the file into list
        to properly read the path of the audio file
        """
        # print('OpenFunction not in use yet')
        self.x.start()
        print('Parallel process for data collection started')

    def playB(self):
        """
        play the file
        """
        self.timer.start(0.01) # 1 / (2xFs) --> 2ms

    def stopB(self):
        """
        play the audio file
        """
        print('Stop')
        # self.x.join()
        # self.board.disconnect()
        self.timer.stop()
        self.reset_bufers()

    # ------------------------------------------------------------------------
    # Plot Configuration
    # ------------------------------------------------------------------------
    def _configure_plot(self):
        """
        Configures specific elements of the PyQtGraph plots.
        :return:
        """
        # self.mainW = pg.GraphicsWindow(title='Spectrogram', size=(800,600))

        # QMainWindow.setObjectName("AID")
        #        self.label = pg.LabelItem(justify='right')
        #        _mainWindow.addItem(self.label)
        self.plt1.setBackground(background=None)
        self.plt1.setAntialiasing(True)
        self._plt1 = self.plt1.addPlot(row=1, col=1)
        self._plt1.setLabel('bottom', "Tiempo", "s")
        self._plt1.setLabel('left', "Amplitud", "Volt")
        self._plt1.showGrid(x=False, y=False)
        # self._plt1.setYRange(-200, 200)
        self._plt1.setXRange(0, int(FS*DT))
        self._plt1.enableAutoRange('y', True)
        self._plt1.enableAutoRange('x', True)

        self.plt2.setBackground(background=None)
        self.plt2.setAntialiasing(True)
        self._plt2 = self.plt2.addPlot(row=1, col=1)
        self._plt2.setLabel('bottom', "Tiempo", "s")
        self._plt2.setLabel('left', "Amplitud", "Volt")
        self._plt2.showGrid(x=False, y=False)
        self._plt2.setYRange(0, 6)
        self._plt2.setXRange(0, int(FS*DT))
        self._plt2.enableAutoRange('x', True)
        self._plt2.enableAutoRange('y', True)

    # ------------------------------------------------------------------------
    # Process - Updates
    # ------------------------------------------------------------------------

    def timer_1(self):
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self._update_plot)

    def reset_bufers(self):
        self.t = 0
        self.d_signal = self.d_signal * 0


def print_raw(sample):
    global Data
    Data.append([i * uVolts_per_count for i in sample.channels_data])
    # print("Data: %f, Lenght: %i " % (sample.channels_data[0], len(sample.channels_data)))
    # 1st plot roll down one and replace leading edge with new data
    # Data = np.array(sample.channels_data[0] * uVolts_per_count) / 400000
    #print(Data)


def start_board():
    board = OpenBCICyton(port='/dev/ttyUSB0', daisy=False)
    board.start_stream(print_raw)


if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # Instancia para iniciar una aplicacion en windows
        app = QApplication(sys.argv)
        # debemos crear un objeto para la clase creada arriba
        _mainWindow = mainWindow()
        # muestra la ventana
        _mainWindow.show()
        # ejecutar la aplicacion
        app.exec_()