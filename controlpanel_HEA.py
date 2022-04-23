import sarah1 as sa
import wfdb

datasets = open('dsets.txt', 'r')
row = datasets.read().split('\n')
row = row[0:-1]

for id in row:

    # Define path to data
    file = "LTAF_84P/" + id

    # Read data from binary files
    data = wfdb.rdrecord(file)
    # Use the command display(data.__dict__) to show dictionary

    # Assign ECG signals
    ecgs = data.p_signal

    # Determine the sampling frequency
    f0 = data.fs

    # Read only one ECG channel
    ECG = ecgs[:, 0]

    # Filter ECG signal via Butterworth
    ECG = sa.filterECG(ECG, f0)

    """
        --------------
        controlpanel()

        Parameters:
            ecg,
            sampling_rate,
            start,             # In seconds
            time_width,        # In minutes
            n_iterations=None, # Set None for a complete scanning
            patient,
            getrpeaks=False,   # For debugging ->
            showdata=False,
            bandwidth=2

        --------------
        cpanel2()

        Parameters:
            ecg,
            sampling_rate,
            start,             # In seconds
            time_width,        # In seconds
            n_iterations=None, # Set None for a complete scanning
            patient,
            getrpeaks=False,   # For debugging ->
            showdata=False,
            bandwidth=2)

    """

    sa.cpanel2(ECG, f0, 30*60, 60, 1, 'LTAF_84P-' + id)

datasets.close()
