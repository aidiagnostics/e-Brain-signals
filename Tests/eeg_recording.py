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
# Multiply uVolts_per_count to convert the channels_data to uVolts.
uVolts_per_count = (4500000)/24/(2**23-1) #uV/count
# Multiply accel_G_per_count to convert the aux_data to G
Volts_per_count = 1.2 * 8388607.0 * 1.5 * 51.0 #V/count
# Time setup
t_sec = 5
t_end = time.time() + t_sec
# Connecting to board
board = OpenBCICyton(port='/dev/ttyUSB0', daisy=False)
#%%
while True:
    board.start_stream(print_raw)
    if time.time() < t_end:
        break