# Reference
# 1. https://github.com/kevinmgamboa/Plotting_OpenBCI_Cyton_Data_live
from pyOpenBCI import OpenBCICyton
import time
import threading
#%%
# --------------------
# Functions Needed
# --------------------
def print_raw(sample):
    print("Data: %f, Lenght: %i " %(sample.channels_data[0], len(sample.channels_data)))
#%%
# --------------------
# Setup
# --------------------
def start_cyton():
    try:
        board.start_stream(print_raw)
    except:
        pass

#%%

# Connecting to board
board = OpenBCICyton(port='/dev/ttyUSB0', daisy=False)

y = threading.Thread(target=start_cyton)
y.daemon = True
# First trial, starts board,wait 5 sec and stops streaming
y.start()
time.sleep(10)
board.stop_stream()
print('finished')