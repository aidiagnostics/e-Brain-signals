"""
Updated on Wed Feb 26 10:10:54 2020

Personal Processing Functions 1 (ppfunctions_1)
Created on Mon Apr  9 11:48:37 2018
@author: Kevin MAchado

Module file containing Python definitions and statements
"""

# Libraries

import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, butter, filtfilt, bessel
from scipy.fftpack import fft
from scipy import signal as sg
from tqdm import tqdm
#import peakutils                                # Librery to help in peak detection

# Functions 
# Normal energy = np.sum(pcgFFT1**2)
# Normalized average Shannon Energy = sum((l**2)*np.log(l**2))/l.shape[0]
# Third Order Shannon Energy = sum((l**3)*np.log(l**3))/l.shape[0]
# -----------------------------------------------------------------------------
#                           Statistic Variables
# -----------------------------------------------------------------------------
def _meanDC(x):
    """
    _meanDC calculates the mean value of the vector (average value)
    """
    miu = 0.0
    for i in range(x.shape[0]):
        miu += x[i]
    miu = miu/len(x)
    return miu
        
def _variance(x):
    """
    _variance It is interpreted also as the power of the fluctuation
    of the signal with respect to the mean
    """
    vari = 0.0
    # mean value
    miu = _meanDC(x)
        
    for i in range(x.shape[0]):
        vari = vari + (x[i]-miu)**2
    vari = vari/len(x)
    return vari

def _StandarD(x):

    vari = 0.0
    vari = _variance(x)
    _Std = vari **(.5)
    return _Std

def _CV(x):
    # Coefficient of Variation
    var = _variance(x)
    std = _StandarD(x)
    
    return 100*(var/std)

def meanDeviation_v(x):
    """
    meanDeviation_v removes the negatives values from the signal
    """
    mdv = np.zeros(len(x))          # mdv stated for "mean deviation vector"
    vdv = mdv                       # vdv stated for "variance deviation vector"
    sDdv = mdv                      # sDdv "Standar Deviation Deviation Vecto"
    
    miu = _meanDC(x)
    var = _variance(x)
    std = _StandarD(x)
    
    for i in range(x.shape[0]):
        mdv[i] = abs(x[i]-miu)
        vdv[i] = abs(x[i]-var)
        sDdv[i] = abs(x[i]-std)
        
    return mdv, vdv, sDdv

def stat_vectors(x):
    '''
    This function calculates a mean vector, variance vector and 
    standar deviation vector at each point of x
    '''
    # Mean
    mi = _meanDC(x)
    miu = np.zeros(len(x))
    var = miu
    std = miu
    l = len(x)
    for i in range(x.shape[0]):
        miu[i] += x[i]/l
        var[i] = (1/l) * (x[i]-mi)**2
        std[i] = (1/l) * (x[i]-mi)**2
    std = std ** (.5)
    # Variance
    return vec_nor(miu), vec_nor(var), vec_nor(std)

def stat_vectors_2(x):
    # Mean
    mi = _variance(x)
    miu = np.zeros(len(x))
    var = miu
    std = miu
    l = len(x)
    for i in range(x.shape[0]):
        miu[i] += x[i]/l
        var[i] = (1/l) * (x[i]-mi)**2
        std[i] = (1/l) * (x[i]-mi)**2
    std = std ** (.5)
    # Variance
    return vec_nor(miu), vec_nor(var), vec_nor(std)
# -----------------------------------------------------------------------------
# PDS
# -----------------------------------------------------------------------------
def vec_nor(x):
    """
    Normalize the amplitude of a vector from -1 to 1
    """
    nVec = np.zeros(len(x));		   # Initializate derivate vector
    nVec = np.divide(x, max(x))
    nVec = nVec-np.mean(nVec);
    nVec = np.divide(nVec,np.max(nVec));
        
    return nVec

def normalize_vector(vector, target_max,  target_min):
    """
    Normalize the amplitude of a vector a min and max value.
    The function doesn't return the signal in zero based line
    """
    mean = np.mean(vector)
    std = np.std(vector)
    vector_zmc = (vector-mean)/std
    xmax = np.amax(vector_zmc)
    xmin = np.amin(vector_zmc)
    d = (vector_zmc - xmax)*((target_max - target_min)/(xmax - xmin)) + target_max      
    return d

def running_sum(x):
    """
    Running Sum Algorithm of an input signal is y[n]= x[n] + y[n-1] 
    """
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i] + y[i-1]
        
    return vec_nor(y)

def derivate_1 (x):
    """
    Derivate of an input signal as y[n]= x[n] - x[n-1] 
    """
    y=np.zeros(len(x));				# Initializate derivate vector
    for i in range(len(x)):
        y[i]=x[i]-x[i-1];		
    return vec_nor(y)
        
def derivate (x):
    """
    Derivate of an input signal as y[n]= x[n+1]- x[n-1] 
    """
    lenght=x.shape[0]				# Get the length of the vector
    y=np.zeros(lenght);				# Initializate derivate vector
    for i in range(lenght-1):
        y[i]=x[i-1]-x[i];		
    return y

def derivate_positive (x):
    """
    Derivate of an input signal as y[n]= x[n+1]- x[n-1] 
    for all values where the signal is positive
    """
    lenght=x.shape[0]				# Get the length of the vector
    y=np.zeros(lenght);				# Initializate derivate vector
    for i in range(lenght-1):
        if x[i]>0:
            y[i]=x[i-1]-x[i];		
    return y
# -----------------------------------------------------------------------------
# Energy
# -----------------------------------------------------------------------------
def Energy_value (x):
    """
    Energy of an input signal  
    """
    y = np.sum(x**2)
    return y

def shannonE_value (x):
    """
    Shannon energy of an input signal  
    """
    y = sum((x**2)*np.log(x**2))/x.shape[0]
    return y

def shannonE_vector (x):
    """
    Shannon energy of an input signal  
    """
    mu = -(x**2)*np.log(x**2)/x.shape[0]
    y = -(((x**2)*np.log(x**2)) - mu)/np.std(x)
    return y

def shannonE_vector_1 (x):
    """
    Shannon energy of an input signal  
    """ 
#    Se = -(1/N) * 
    mu = -(x**2)*np.log(x**2)/x.shape[0]
    y = -(((x**2)*np.log(x**2)) - mu)/np.std(x)
    return y

def E_VS_LF (pcgFFT1, vTfft1, on):
    """
    Energy of PCG Vibratory Spectrum in low Frequencies (E_VS_LF)
    (frequency components, frequency value vector, on = on percentage or not)
    According with [1] The total vibratory spectrum can be divided into 7 bands.
    This is a modification of this 7 bands
    1. 0-5Hz, 2. 5-25Hz; 3. 25-60Hz; 4. 60-120Hz; 5. 120-400Hz

The PCG signal producess vibrations in the spectrum between 0-2k Hz. 
[1] Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.
    """
    c1 = (np.abs(vTfft1-5)).argmin()
    c2 = (np.abs(vTfft1-25)).argmin()
    c3 = (np.abs(vTfft1-120)).argmin()
    c4 = (np.abs(vTfft1-240)).argmin()
    c5 = (np.abs(vTfft1-500)).argmin()
    
    # All vector energy
    xAll = Energy_value(pcgFFT1)

    # Procesando de 0.01-5 Hz
    pcgFFT_F1 = pcgFFT1[0:c1]
    x1 = Energy_value(pcgFFT_F1)
    
    # Procesando de 5-25 Hz
    pcgFFT_F2 = pcgFFT1[c1:c2]
    x2 = Energy_value(pcgFFT_F2)
    
    # Procesando de 25-120 Hz
    pcgFFT_F3 = pcgFFT1[c2:c3]
    x3 = Energy_value(pcgFFT_F3)
    
    # Procesando de 120-240 Hz
    pcgFFT_F4 = pcgFFT1[c3:c4]
    x4 = Energy_value(pcgFFT_F4)
    
    # Procesando de 240-500 Hz
    pcgFFT_F5 = pcgFFT1[c4:c5]
    x5 = Energy_value(pcgFFT_F5)
    
    x = np.array([xAll, x1, x2, x3, x4, x5])
    
    if (on == 'percentage'):
        x = 100*(x/x[0])

    return x

def E_VS (pcgFFT1, vTfft1, on):
    """
    Energy of PCG Vibratory Spectrum
    (frequency components, frequency value vector, on = on percentage or not)
    According with [1] The total vibratory spectrum can be divided into 7 bands:
    1. 0-5Hz, 2. 5-25Hz; 3. 25-120Hz; 4. 120-240Hz; 5. 240-500Hz; 6. 500-1000Hz; 7. 1000-2000Hz

The PCG signal producess vibrations in the spectrum between 0-2k Hz. 
[1] Abbas, Abbas K. (Abbas Khudair), Bassam, Rasha and Morgan & Claypool Publishers Phonocardiography signal processing. Morgan & Claypool Publishers, San Rafael, Calif, 2009.
    """
    c1 = (np.abs(vTfft1-5)).argmin()
    c2 = (np.abs(vTfft1-25)).argmin()
    c3 = (np.abs(vTfft1-120)).argmin()
    c4 = (np.abs(vTfft1-240)).argmin()
    c5 = (np.abs(vTfft1-500)).argmin()
    c6 = (np.abs(vTfft1-1000)).argmin()
    c7 = (np.abs(vTfft1-2000)).argmin()
    
    # All vector energy
    xAll = Energy_value(pcgFFT1)

    # Procesando de 0.01-5 Hz
    pcgFFT_F1 = pcgFFT1[0:c1]
    x1 = Energy_value(pcgFFT_F1)
    
    # Procesando de 5-25 Hz
    pcgFFT_F2 = pcgFFT1[c1:c2]
    x2 = Energy_value(pcgFFT_F2)
    
    # Procesando de 25-120 Hz
    pcgFFT_F3 = pcgFFT1[c2:c3]
    x3 = Energy_value(pcgFFT_F3)
    
    # Procesando de 120-240 Hz
    pcgFFT_F4 = pcgFFT1[c3:c4]
    x4 = Energy_value(pcgFFT_F4)
    
    # Procesando de 240-500 Hz
    pcgFFT_F5 = pcgFFT1[c4:c5]
    x5 = Energy_value(pcgFFT_F5)
    
    # Procesando de 500-1000 Hz
    pcgFFT_F6 = pcgFFT1[c5:c6]
    x6 = Energy_value(pcgFFT_F6)
    
    # Procesando de 1000-2000 Hz
    pcgFFT_F7 = pcgFFT1[c6:c7]
    x7 = Energy_value(pcgFFT_F7)
    
    x = np.array([xAll, x1, x2, x3, x4, x5, x6, x7])
    
    if (on == 'percentage'):
        x = 100*(x/x[0])

    return x
# -----------------------------------------------------------------------------
#                                Envelopes
# -----------------------------------------------------------------------------
def sim_envelope(x, fs, seconds=0.3, smooth = [False, 10]):
    """
    Develope by: Kevin Machado G
    sim_envelope states for "simple envelope"
        Reference
    C. Jarne. Simple empirical algorithm to obtain signal envelope in three steps
    March 21, 2017. University of Quilmes (UNQ) e-mail: cecilia.jarne@unq.edu.ar
    """
    Y = np.zeros(len(x))
    # 1. Take the absolute value of the signal
    SE_x = abs(x)
    # 2. Divide the signal into k bunches of N samples (corresponding to 0.1 s) (0.1*sf)/1
    sec = seconds
    N = int(sec*fs)
    k, m = divmod(len(x), N)
    
    for i in range(N):
        n = max(SE_x[i*k:i*k+(k-1)])
        Y[i*k:i*k+k] = n
    if smooth[0] == True:
    # 4. Smoothing the signal
        Y = butter_lowpass_filter(Y, smooth[1], fs, order = 1)
    return Y

#-------------------------------------------
def features_1(data_P1, fs):
    
    # Defining the Vibratory Frequency Bands
    bVec = [0.01, 120, 240, 350, 425, 500, 999]
    # Initializing Vectors
    band_matrix = np.zeros((len(bVec),len(data_P1))) # Band Matrix
    power_matrix = np.zeros((len(bVec),int(1+(len(data_P1)/2)))) # Band Matrix
    freqs_matrix = np.zeros((len(bVec),int(1+(len(data_P1)/2)))) # Band Matrix
    SePCG = np.zeros(len(bVec)-1)                            # Shannon Energy in each Band
    SePWR = np.zeros(len(bVec)-1)                            # Shannon Energy in each Band
      
    
    for i in range(len(bVec)-1):
        band_matrix[i,:] = butter_bp_fil(data_P1, bVec[i], bVec[i+1], fs)       
        freqs_matrix[i,:], power_matrix[i,:] = sg.periodogram(band_matrix[i,:], fs, scaling = 'density')
        SePCG[i] = Energy_value(band_matrix[i,:])
        SePWR[i] = Energy_value(power_matrix[i,:])
     
    SePWR = 1*np.log10(SePWR)
    SePCG = 1*np.log10(SePCG)
    
    return SePWR, SePCG

# -----------------------------------------------------------------------------
# Filter Processes
# -----------------------------------------------------------------------------
    
def recursive_moving_average_F(X, Fs, M):
    """
    The recursive Moving Average Filter its an algorithm to implement the typical
    moving average filter more faster. The algorithm is written as:
    y[i] = y[i-1] + x[i+p] - x[i-q]
    p = (M - 1)/2,   q = p + 1
    Ref: Udemy Course "Digital Signal Processing (DSP) From Ground Up™ in Python", 
    available in: https://www.udemy.com/python-dsp/
    ---------------------------------------------------------------------------
    X: input signal
    Fs: Sampling Frequency
    M: number of points in M range of time the moving average
    
    """
    p = int(((M*Fs)-1)/2)
    q = p + 1
    Y = np.zeros(len(X))
    for i in range(len(X)):
        Y[i] = Y[i-1] + X[i-p] - X[i-q]
        
    return vec_nor(Y)

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
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bp_fil(data, lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter
    Ref: 
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    b, a = butter_bp_coe(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return vec_nor(y)


def Fpass(X,lp):
    """
    Fpass is the function to pass the coefficients of a filter trough a signal'
    """
    llp=np.size(lp)	  	        # Get the length of the lowpass vector		

    x=np.convolve(X,lp);		        # Disrete convolution 
    x=x[int(llp/2):-int(llp/2)];
    x=x-(np.mean(x));
    x=x/np.max(x);
    
    y=vec_nor(x);				# Vector Normalizing
        
    return y

def FpassBand(X,hp,lp):
    """
    FpassBand is the function that develop a pass band filter of the signal 'x' through the
    discrete convolution of this 'x' first with the coeficients of a High Pass Filter 'hp' and then
    with the discrete convolution of this result with a Low Pass Filter 'lp'
    """
    llp=lp.shape[0]	  	            # Get the length of the lowpass vector		
    lhp=hp.shape[0]			        # Get the length of the highpass vector		

    x=np.convolve(X,lp);		        # Disrete convolution 
    x=x[int(llp/2):-int(llp/2)];
    x=vec_nor(x)
	
    y=np.convolve(x,hp);			    # Disrete onvolution
    y=y[int(lhp/2):-int(lhp/2)];
    y=vec_nor(y)

    return y

def FpassBand_1(X, Fs, f1, f2):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FpassBand_1 is a function to develop a passband filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """

    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # 
    taps = firwin(100, [f1, f2], pass_zero=False)
#    taps = firwin(N, L_cutoff_hz/nyq_rate, window=('kaiser', beta))
#    taps_2 = firwin(N, H_cutoff_hz/nyq_rate, pass_zero=True)
    # Use lfilter to filter x with the FIR filter.
    X_pb= lfilter(taps, 1.0, X)
   # X_pb= lfilter(taps_2, 1.0, X_l)
    
    return X_pb[N-1:]

def FpassBand_2(X, f1, f2, Fs):
    
    Y = FhighPass(X, Fs, f1)
    
    Y = FlowPass(X, Fs, f2)
    
    return Y

def FhighPass(X, Fs, H_cutoff_hz):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FhighPass is a function to develop a highpass filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps_2 = firwin(N, H_cutoff_hz/nyq_rate, pass_zero=False)
    # Use lfilter to filter x with the FIR filter.
    X_h= lfilter(taps_2, 1.0, X)
    
    return X_h[N-1:]
    
def FlowPass(X, Fs, L_cutoff_hz):
    """
    Ref: http://scipy-cookbook.readthedocs.io/items/FIRFilter.html
    http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.signal.firwin.html
    FlowPass is a function to develop a lowpass filter using 'firwin'
    and 'lfilter' functions from the "Scipy" library
    """
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, L_cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    X_l= lfilter(taps, 1.0, X)
    
    return X_l[N-1:]
# -----------------------------------------------------------------------------
def bessel_bp_coe(lowcut, highcut, fs, order=4):
    """
    For reference check the Bessel Python Function
    """
#    nyq = (2*np.pi)/sf
#    low = lowcut*nyq
#    high = highcut*nyq
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = bessel(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bessel_bp_fil(data, lowcut, highcut, fs, order=4):
    """
    For reference check the Bessel Python Function
    """
    b, a = bessel_bp_coe(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return vec_nor(y)
# -----------------------------------------------------------------------------
def cheb1_bp_coe(lowcut, highcut, fs, order=4):
    """
    For reference check the Chebyshev Python Function
    """
#    nyq = (2*np.pi)/sf
#    low = lowcut*nyq
#    high = highcut*nyq
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sg.cheby1(order, 1, [low, high], btype='bandpass', analog=False)
    return b, a

def cheb1_bp_fil(data, lowcut, highcut, fs, order=4):
    """
    For reference check the Chebyshev Python Function
    """
    b, a = cheb1_bp_coe(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return vec_nor(y)
# -----------------------------------------------------------------------------
def butter_lowpass(cutoff, fs, order = 3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order = 3):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order = 3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order = 3):
    b, a = butter_highpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y
# -----------------------------------------------------------------------------
#                               Signal Spectrogram
# -----------------------------------------------------------------------------
def my_spec(X, fft_size = 80, step_size = 1, spec_thresh = 2.5):
    ### Short-Time Fourier Transform ###
    spec = stft(X.astype('float32'), fftsize=fft_size, step=step_size, real=False,compute_onesided=True)
    ### Extract Real values from Complex Numbers ###
    spec = abs(spec)
    ### Max 1 normalization ###
    spec /= spec.max()
    ### Log ###
    spec = np.log10(spec)
    ### Threshold ###
    spec[spec < -spec_thresh] = -spec_thresh
    ### Signal Transpose ###
    spec = np.transpose(spec)
    
    return spec

def xrange(x):
    return iter(range(x))

def realtime_spectrogram(data, sf):
    """
    Converts raw chunks into spectrogram chunks.

    Parameters
    ----------
    data : Tuple
        Raw data object to be windowed.
    """
    # Spectrogram parameters
    fft_size = int(sf / 2)  # 1000 samples represent 500ms in time  # window size for the FFT
    step_size = 1  # distance to slide along the window (in time) if devided by 40 is good
    spec_thresh = 5  # threshold for spectrogram (lower filters out more noise)

    # Initialising chunked-spectrograms variable
    X = []
    for sample_idx in tqdm(range(len(data))):
        ch1 = aid_spectrogram(data[sample_idx][0].astype('float64'), log=True, thresh=spec_thresh, fft_size=fft_size,
                              step_size=step_size)
        ch2 = aid_spectrogram(data[sample_idx][1].astype('float64'), log=True, thresh=spec_thresh, fft_size=fft_size,
                              step_size=step_size)
        chs = np.dstack((ch1, ch2))

        X.append(chs)

    print("Finished spectrogram transformation")

    return X

def aid_spectrogram(d,log = True, thresh= 5, fft_size = 512, step_size = 64, window_type=0):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False, 
                           compute_onesided=True, window_type=window_type))
  
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return np.transpose(specgram)

def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw,ws),dtype = a.dtype)
    i=0

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out

def overlapping(signal, stepsize=1, fs=1000, time_window=0.2, number_time_samples=1):
    """
    Create an overlapped version of the input signal
    Parameters
    ----------
    signal : input vector corresponding to audio data with shape (n_audio_poins,)
    fs : sampling rate of the input signal
    time_window : Duration of the window in seconds
    stepsize : number of samples between the beginning of a window and the beginning of the next one

    Returns
    -------
    chunked_and_overlapped_signal : Matrix with shape (number_examples, number_time_samples)
    """
    if number_time_samples == 1:
        number_time_samples = int(fs*time_window)  
    else:
        pass
    append = np.zeros((number_time_samples - len(signal) % number_time_samples))  #this calculates how many samples to add to the input vector for the windows to fit along it, and creates a zeros vector of that size.  
    signal = np.hstack((signal, append)) #completes the input vector with the zeros vector created in order to have an even number of windows fit in the data
    result = np.vstack( signal[i:i+number_time_samples] for i in range(0,len(signal)-number_time_samples, stepsize))
    return result

def stft(X, fftsize=40, step=66, mean_normalize=True, real=False, compute_onesided=True, window_type=0):
    """
    Compute Short-Time Fourier Transform (STFT) for 1D real valued input X
    fftsize 80 samples corresponding to 40ms
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlapping(X, stepsize=step, number_time_samples=fftsize)
    
    size = fftsize
    if window_type==0:
        win = sg.general_gaussian(size, p=0.5, sig=200) 
    if window_type==1:
        win = 0.8 - .9 * np.sin(2 * np.pi * np.arange(size) / (size - 1)) # Modify Hamming Window
    if window_type==2:
        win = 100 - 1 * np.tanh(2 * np.pi * np.arange(size) / (size - 1))    #Hann window 
    if window_type==3:
        minR = 0
        ampR = 1.0814
        f = 2000
        win = minR + ampR * (0.05*np.sin(2*np.pi*np.arange(size)*3*f)
        + 0.4*np.sin(2*np.pi*np.arange(size)*f) + 0.25*np.sin(2*np.pi*np.arange(size)*2*f+45))
        
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X

def move_spec(spectrogram, shift = 100):
    '''This function move the spectogram 'shift' numbers of points to the right.
    This is done to fix the displacement given when applying the spectrogram with
    our personal function'''
    cut = spectrogram[:,0:shift]
    for i in range(spectrogram.shape[1] -1, spectrogram.shape[1] - shift, -1):
        spectrogram = np.roll(spectrogram, 1, axis=1)
        spectrogram[:, -1] = 0
    spectrogram[:,0:shift] = cut
    return spectrogram
# -----------------------------------------------------------------------------
           # Segmentation By Running Sum Algorithm & Filters
# -----------------------------------------------------------------------------
def seg_RSA1(x, fs):
    # # Appliying Running Sum Algorithm of PCG filtered from 0.01Hz to 1kHz
    F_x = running_sum(vec_nor(butter_bp_fil(x, 0.01, 50, fs)))
    # Smoothing the signal by filtering from 0.01Hz to 5Hz
    F_x = butter_bp_fil(F_x,0.01, 2, fs)
    # Appliying 1st derivative to indentify slope sign changes
    xx = derivate_1(F_x)
    #
    xx = butter_bp_fil(xx, 0.01,2, fs)
    # Transforming positives to 1 & negatives to -1
    xxS = np.sign(xx)
    
    return xxS, F_x
# -----------------------------------------------------------------------------
                   # Segmentation By Derivatives
# -----------------------------------------------------------------------------                                
def seg_Der1(x, fs):
    # Segmenting in an specific frequency band
    pcgPeaks, peaks, allpeaks = PDP(x, fs)
    # -----------------------------------------------------------------------------
    # Time processing
    dT = 0.4          # [1] mean S1 duration 122ms, mean S2 duration 92ms
    timeV = []
    timeV.append(0)
    pointV = []
    pointV.append(0)
    segmV = np.zeros(len(x))
    
    for i in range(len(pcgPeaks)-1):
        if pcgPeaks[i]>0.5:
            timeV.append(i/fs)       # Gives the time when a peak get found
            pointV.append(i)          # Gives the time when a peak get found
            if (pointV[-1]/fs)-(pointV[-2]/fs)> dT:
                segmV[pointV[-2]:pointV[-1]] = 0.4  # Marc a diastolic segment
            else:
                segmV[pointV[-2]:pointV[-1]] = 0.6  # Marc a systolic segment
    
    return segmV, pcgPeaks

def seg_Der2(x, fs):
    # Appliying Running Sum Algorithm of PCG filtered from 0.01Hz to 1kHz
    F_x = running_sum(vec_nor(butter_bp_fil(x, 0.01, 999, fs)))
    # Smoothing the signal by filtering from 0.01Hz to 5Hz
    F_x = butter_bp_fil(F_x,0.01, 5, fs)
    # Time to be represented in samples
    time_samples = 0.5
    # Number of samples to move over the signal
    mC = int(time_samples * fs)                       
    peaks = np.zeros(len(F_x))
    p = sg.find_peaks(F_x, distance=mC)
    # Defining peaks as +1
    for i in range (len(p[0][:])):
        peaks[p[0][i]] = 1
    
    return peaks                     
# -----------------------------------------------------------------------------
                                # Peak Detection
# -----------------------------------------------------------------------------
def PDP(Xf, samplerate):
    """
    Peak Detection Process
    """
    timeCut = samplerate*0.25                      # Time to count another pulse
    vCorte = 0.6                                   # Amplitude threshold
    
    Xf = vec_nor(Xf)                               # Normalize signal
    dX = derivate_positive(Xf);				      # Derivate of the signal
    dX = vec_nor(dX);			                  # Vector Normalizing
    
    size=np.shape(Xf)				                 # Rank or dimension of the array
    fil=size[0];					                     # Number of rows
 
    positive=np.zeros((1,fil+1));                   # Initializating Vector 
    positive=positive[0];                           # Getting the Vector

    points=np.zeros((1,fil));                       # Initializating the Peak Points Vector
    points=points[0];                               # Getting the point vector

    points1=np.zeros((1,fil));                      # Initializating the Peak Points Vector
    points1=points1[0];                             # Getting the point vector
       
    '''
    FIRST! having the positives values of the slope as 1
    And the negative values of the slope as 0
    '''
    for i in range(0,fil):
        if Xf[i] > 0:
            if dX[i]>0:
                positive[i] = Xf[i];
            else:
                positive[i] = 0;
    '''
    SECOND! a peak will be found when the ith value is equal to 1 &&
    the ith+1 is equal to 0
    '''
    for i in range(0,fil):
        if (positive[i]==Xf[i] and positive[i+1]==0):
            points[i] = Xf[i];
        else:
            points[i] = 0;
    '''
    THIRD! Define a minimun Peak Height
    '''
    p=0;
    for i in range(0,fil):
        if (Xf[i] > vCorte and p==0):
            p = i
            points1[i] = Xf[i]
        else:
            points1[i] = 0
            if (p+timeCut < i):
                p = 0
                    
    return points1, points, positive[0:(len(positive)-1)]
# -----------------------------------------------------------------------------
# Peak Detection 2
# -----------------------------------------------------------------------------
def PDP_2(Xf, samplerate):
    """
    Peak Detection Process
    """
    timeCut = samplerate*0.25                      # Time to count another pulse
    vCorte = 0.6                                   # Amplitude threshold
    
    #Xf = vec_nor(Xf)                               # Normalize signal
    dX = np.diff(Xf);				      # Derivate of the signal
    dX = vec_nor(dX);			                  # Vector Normalizing
    
    size=np.shape(Xf)				                 # Rank or dimension of the array
    fil=size[0];					                     # Number of rows
 
    positive=np.zeros((1,fil+1));                   # Initializating Vector 
    positive=positive[0];                           # Getting the Vector

    points=np.zeros((1,fil));                       # Initializating the Peak Points Vector
    points=points[0];                               # Getting the point vector

    points1=np.zeros((1,fil));                      # Initializating the Peak Points Vector
    points1=points1[0];                             # Getting the point vector
       
    '''
    FIRST! having the positives values of the slope as 1
    And the negative values of the slope as 0
    '''
    for i in range(0,fil-1):
        if Xf[i] > 0:
            if dX[i]>0:
                positive[i] = Xf[i];
            else:
                positive[i] = 0;
    '''
    SECOND! a peak will be found when the ith value is equal to 1 &&
    the ith+1 is equal to 0
    '''
    for i in range(0,fil):
        if (positive[i]==Xf[i] and positive[i+1]==0):
            points[i] = Xf[i];
        else:
            points[i] = 0;
    '''
    THIRD! Define a minimun Peak Height
    '''
    p=0;
    for i in range(0,fil):
        if (Xf[i] > vCorte and p==0):
            p = i
            points1[i] = Xf[i]
        else:
            points1[i] = 0
            if (p+timeCut < i):
                p = 0
                    
    return points1, points, positive[0:(len(positive)-1)]
# -----------------------------------------------------------------------------
                              # Fast Fourier Transform
# -----------------------------------------------------------------------------
def fft_k(data, samplerate, showFrequency):
    '''
    Fast Fourier Transform using fftpack from scipy
    Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    '''
    # FFT Full Vector 'k' coefficients
    pcgFFT = fft(data)
    # FFT positives values                                                    
    short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[0:(np.size(data)//2)])
    #short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[(np.size(data)//2):None])  vector selected from the middle to the end
    # Vector of frequencies (X-axes)
    vTfft = np.linspace(0.0, 1.0/(2.0*(1/samplerate)), np.size(data)//2)  
    # find the value closest to a value   
    idx = (np.abs(vTfft-showFrequency)).argmin()             
    
    return short_pcgFFT[0:idx], vTfft[0:idx]

def fft_k_N(data, samplerate, showFrequency):
    '''
    Normalized Fast Fourier Transform
    Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    '''
    # FFT Full Vector 'k' coefficients
    pcgFFT = fft(data)
    # FFT positives values from the middle to the end (to evoid the interference at beginning)
    short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[0:(np.size(data)//2)])
    #short_pcgFFT = 2.0/np.size(data) * np.abs(pcgFFT[(np.size(data)//2):None])  vector selected from the middle to the end
    # Vector of frequencies (X-axes)
    vTfft = np.linspace(0.0, 1.0/(2.0*(1/samplerate)), np.size(data)//2)  
    # find the value closest to a value   
    idx = (np.abs(vTfft-showFrequency)).argmin()            
    
    return vec_nor(short_pcgFFT[0:idx]), vTfft[0:idx]


# -----------------------------------------------------------------------------
                              # PCG Audio Pre-Processing
# -----------------------------------------------------------------------------
def pre_pro_audio_PCG(x, fs):
# Ensure having a Mono sound
    if len(x.shape)>1:
    # select the left size
        x = x[:,0]
    # Resampling Audio PCG to 2k Hz
    Frs = 2000
    Nrs = int(Frs*(len(x)/fs)) # FsC  N; Fsn  x
    if fs > Frs:
        x = sg.resample(x, Nrs)
    
    return vec_nor(x), Frs
# -----------------------------------------------------------------------------
#             Pre-Process: Signal basic information (SBI)
# -----------------------------------------------------------------------------
def pre_pro_basicInfo_PCG(x, fs):
    # find the time duration of the sound
    t_sound = 1/fs*len(x)
    # make a vector time for the sound
    vec_time = np.linspace(0, t_sound, len(x))
    return t_sound, vec_time