# Importing libraries
import mne
import numpy as np
from tqdm import tqdm


# Loading EEG signals
signal = mne.io.read_raw('Tests/ane_SD_EMG_1010_SED_1_raw.fif', preload=True)
# sampling frequency
sf = signal.info['sfreq']
# Extracting two channels
ch = signal._data[:2, :]
# %%
# ------------------------------------------------------------------------------------
#                            Implementing realtime Spectrogram
# ------------------------------------------------------------------------------------
# samples in 3 sec
samples = int(sf*3)
# Number of epochs in 3 seconds
nep = int(ch.shape[1]/samples)

#%%
# Loop where signal is transformed into spectrogram
for n in tqdm(range(nep)):
    print([ch[0][n*samples:(n+1)*samples], ch[1][n*samples:(n+1)*samples]])

