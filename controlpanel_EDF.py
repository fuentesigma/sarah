import sarah1 as sa
import pyedflib as edf

# Patient number
patient = "200003"

# Set path accordingly
path = "../polysomnography/edfs/shhs1/shhs1-"

# Open file
file = edf.EdfReader(path + patient + ".edf")

# Extract data
chn = 3
ECG = file.readSignal(chn)

# Original sampling frequency
f0 = int(file.samplefrequency(chn))

# Filter ECG signal via Butterworth
ECG = -sa.filterECG(ECG, sampling_rate=f0)

# Close file once signals are read
file.close()

# Control panel 2
sa.cpanel2(ECG, f0, 10 * 60, 60, None, 'SSH1-' + patient)
