"""
EEG "Realtime" Visualizer
-------------------------

@author: Kevin Machado Gamboa
Created on Tue Oct 20 22:31:16 2020
References:

"""
import mne
import sys
import numpy as np 
import pyqtgraph as pg

import pyOpenBCI as obci
from pyOpenBCI import OpenBCICyton as cyton

# sys.path.append('./')
# import wavfile
# GUI libraries
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, QtGui
# Own Lobrary
import ppfunctions_1 as ppf

from os.path import dirname as up
mf_path = (up(__file__)).replace('\\','/')    # Main Folder Path

DT = 8    # Display Window lenght in time
UPT = 1   # Update time miliseconds


# def collect_eeg(sample):
#     print("Data: %f, Lenght: %i " % (sample.channels_data[0], (sample.channels_data[0])))

class mainWindow(QMainWindow):
    def __init__(self):
        #Inicia el objeto QMainWindow
        QMainWindow.__init__(self)
        # Loads an .ui file & configure UI
        loadUi("EEG_realtime_ui_v0.ui",self)
        self.OBCI = cyton(port='/dev/ttyUSB0', daisy=False)
        #
#        _translate = QtCore.QCoreApplication.translate
#        QMainWindow.setWindowTitle('AID')
        # Variables
        self.fs = None
        self.data = None
        self.data_P1 = None
        self.t = -8
        self.d_signal = np.zeros(512*DT)                 # Vector Label
        self.duration = None
        self.vectortime = None
        self.file_path = str
        self.plot_colors = ['#ffee3d' , '#0072bd', '#d95319', '#bd0000']
        # Initial sub-functions
        self._configure_plot()
        self.buttons()
        self.openF()
        self.timer_1()

    def collect_eeg(self, sample):
        print("Data: %f, Lenght: %i " % (sample.channels_data[0], (sample.channels_data[0])))

        self._plt1.plot(y=list(sample.channels_data[0]), pen=self.plot_colors[0])

    def _update_plot(self):
        """
        Updates and redraws the graphics in the plot.
        """
        #self.OBCI.start_stream(self.collect_eeg())

        self.duration = np.size(self.data._data[0]) * (1/self.fs)
        self.vectortime = np.linspace(0, self.duration, np.size(self.data._data[0]))

        self.data_P1 = ppf.vec_nor(self.data._data[0])
        
        self.vlabel = np.zeros(len(self.data_P1))
        
        # Xf1 = 1+ppf.butter_bp_fil(self.data_P1, 40, 70, self.fs)
        # Xf2 = 2+ppf.butter_bp_fil(self.data_P1, 70, 100, self.fs)
        # Updating Plots
        self._plt1.clear()
        self._plt1.plot(x=list(self.vectortime), y=list(self.data_P1), pen=self.plot_colors[0])
        #self._plt1.plot(x=list(self.vectortime), y=list(self.vlabel), pen=self.plot_colors[0])

        self._plt2.clear()
        self._plt2.plot(x=list(self.vectortime), y=list(self.data_P1), pen=self.plot_colors[0])
        # self._plt2.plot(x=list(self.vectortime), y=list(Xf1), pen=self.plot_colors[1])
        # self._plt2.plot(x=list(self.vectortime), y=list(Xf2), pen=self.plot_colors[2])

        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        self._plt1.addItem(self.region, ignoreBounds=True)      
        self._plt2.setAutoVisible(y=True)
        
        self._plt2.addItem(self.region2, ignoreBounds=True) 
        
        self._plt2.addItem(self.vLine, ignoreBounds=True)
        self._plt2.addItem(self.hLine, ignoreBounds=True)
        
        self.vb = self._plt2.vb
        self.proxy = pg.SignalProxy(self._plt2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
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
        # self.file_path, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        # # NOTE: 
        # self.data = mne.io.read_raw_edf(self.file_path, preload=True)

        self.data = mne.io.read_raw_edf(mf_path+'/Tests/Test_data/PN00-1.edf', preload=True)
        self.fs = int(self.data.info['sfreq'])
        print('data uploaded', self.data)
    
        print('sampling freq', str(self.fs))
        # plots the signal loaded
        self._update_plot()

    def playB(self):
        """
        play the file 
        """
        print('playing')
        self.OBCI.start_stream(self.collect_eeg)
        #self.timer.start(UPT)

    def stopB(self):
        """
        play the audio file 
        """
        print('Stop')
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
        #self.mainW = pg.GraphicsWindow(title='Spectrogram', size=(800,600))
        
        #QMainWindow.setObjectName("AID")
#        self.label = pg.LabelItem(justify='right')
#        _mainWindow.addItem(self.label)
        self.plt1.setBackground(background=None)
        self.plt1.setAntialiasing(True)
        self._plt1 = self.plt1.addPlot(row=1, col=1)
        self._plt1.setLabel('bottom', "Tiempo", "s")
        self._plt1.setLabel('left', "Amplitud", "Volt")
        self._plt1.showGrid(x=False, y=False)
        
        self.plt2.setBackground(background=None)
        self.plt2.setAntialiasing(True)
        self._plt2 = self.plt2.addPlot(row=1, col=1)
        self._plt2.setLabel('bottom', "Tiempo", "s")
        self._plt2.setLabel('left', "Amplitud", "Volt")
        self._plt2.showGrid(x=False, y=False)
        
        # Region 1 
        self.region = pg.LinearRegionItem()
        self.region.setZValue(1)
        
        self.region.sigRegionChanged.connect(self.update)
        self._plt2.sigRangeChanged.connect(self.updateRegion)
        self.region.setRegion([0, DT])
        # Setting Region 2
        self.region2 = pg.LinearRegionItem()
        self.region2.setZValue(1)
        self.region2.setRegion([0, 0.5])
               
        self.region2.sigRegionChanged.connect(self.update2)
        
        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=[100,100,200,200])
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=[100,100,200,200])
    # ------------------------------------------------------------------------
                             # Process - Updates
    # ------------------------------------------------------------------------    
    
    def dis_update(self):
        # 1st plot roll down one and replace leading edge with new data
        self.d_signal = np.roll(self.d_signal, -self.fs, 0)
        self.d_signal[-self.fs:] = self.data_P1[self.t*self.fs:(1+self.t)*self.fs]
        
        self._plt2.clear()
        self._plt2.plot(x=list(self.vectortime[:self.fs*DT]), y=list(self.d_signal), pen=self.plot_colors[0])
        
        self.t+=1
    
    def update_otro(self):
        self.region.setZValue(1)
        minX, maxX = self.region.getRegion()     # get the min-max values of region
        self._plt2.setXRange(self.t*1, (8+self.t), padding=0)
        self.t+=1
        
        
        
    def update(self):
        self.region.setZValue(1)
        minX, maxX = self.region.getRegion()     # get the min-max values of region
        self._plt2.setXRange(minX, maxX, padding=0)
    
    def updateRegion(self, window, viewRange):
        rgn = viewRange[0]
        self.region.setRegion(rgn)
    
    def update2(self):
        self.region2.setZValue(-10)
        self.minX2, self.maxX2 = self.region2.getRegion()     # get the min-max values of region
#        self.label.setText("<span style='font-size: 12pt'>L1=%0.1f,   <span style='color: red'>L2=%0.1f</span>,   <span style='color: green'>L1-L2=%0.1f</span>" % (self.minX2, self.maxX2, abs(self.minX2-self.maxX2)))
        #self._plt4.setXRange(minX3, maxX3, padding=0)
        
    def updateRegion2(self, window, viewRange):
        rgn_f = viewRange[0]
        self.region3.setRegion(rgn_f)
    
    def mouseMoved(self, evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
        if self._plt2.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            #index = int(mousePoint.x())
            #if index > 0 and index < len(self.data):
            #print(mousePoint.x())
                    #label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            
    def timer_1(self):
        self.timer = pg.QtCore.QTimer()
        #self.timer.timeout.connect(self.dis_update)
        self.timer.timeout.connect(self.update_otro)
        
    def reset_bufers(self):
        self.t = 0
        self.d_signal = self.d_signal*0

        
#Instancia para iniciar una aplicacion en windows
app = QApplication(sys.argv)
#debemos crear un objeto para la clase creada arriba
_mainWindow = mainWindow()
    #muestra la ventana
_mainWindow.show()
    #ejecutar la aplicacion
app.exec_()