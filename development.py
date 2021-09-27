# Importing libraries
import mne
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from helpers_and_functions import utils, main_functions as mf

# %%
# ------------------------------------------------------------------------------------
#                            Prepares the model
# ------------------------------------------------------------------------------------
model_path = 'logs/9685_sleep_20210727_112442_spectrogram_sequential_4/9685_sleep_20210727_112442_spectrogram_sequential_4_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# labels
labels = ['conscious', 'unconscious']
# %%
# ------------------------------------------------------------------------------------
#                            Loads the signal
# ------------------------------------------------------------------------------------
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
samples = int(sf * 3)
# Number of epochs in 3 seconds
nep = int(ch.shape[1] / samples)
# initializing variables
spectro = []
predictions = []
# Loop where signal is transformed into spectrogram
for n in tqdm(range(nep)):
    spectro_transform = mf.realtime_spectrogram(
        np.array([ch[0][n * samples:(n + 1) * samples],
                  ch[1][n * samples:(n + 1) * samples]]), sf)

    prediction = utils.predict_tfl(interpreter, input_details, output_details,
                                   np.expand_dims(spectro_transform, axis=0))

    spectro.append(spectro_transform)
    predictions.append(prediction)

# #%%
# # ------------------------------------------------------------------------------------
# #                            Converting saved model to TFLite
# # ------------------------------------------------------------------------------------
# # model dir
# model_dir = 'logs/9685_sleep_20210727_112442_spectrogram_sequential_4/9685_sleep_20210727_112442_spectrogram_sequential_4_model.h5'
# # load the model
# model = tf.keras.models.load_model(model_dir)
# # tflite converting
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# # Save the model.
# with open('logs/9685_sleep_20210727_112442_spectrogram_sequential_4/9685_sleep_20210727_112442_spectrogram_sequential_4_model.tflite', 'wb') as f:
#   f.write(tflite_model)
