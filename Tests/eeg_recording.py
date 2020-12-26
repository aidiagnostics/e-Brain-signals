
from pyOpenBCI import OpenBCICyton
import time
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
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count
Volts_per_count = 1.2 * 8388607.0 * 1.5 * 51.0 #V/count
# Connecting to board
board = OpenBCICyton(port='/dev/ttyUSB0', daisy=False)
#%%
board.start_stream(print_raw)
time.time(2)
board.disconnect()