"""

    -------------------------- S A R A H 1 -----------------------------

    Statistical Algorithm for the Research and Analysis of Heart Signals

    --------------------------------------------------------------------

    SARAH  V--001   24/12/2021
    Last revision   31/01/2022
    Developed by:   J. Fuentes

    Changelog:
    >> 21/01/22
    .. Last rngb(1, 50)/20 in ecgdists() is now deprecated
    .. New one: rngb(0.5, 40)/20

    >> 25/01/22
    .. Saving figures will now be optional

    >> 30/01/22
    .. Now using sklearn for fitting data

"""
# Library for HDF
import h5py

# Scipy packages
import scipy
import scipy.signal
import scipy.ndimage
import scipy.stats as st

from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Time format
import time

# Basic libraries
import numpy as np
import pandas as pd

# Warnings
from warnings import warn

# Maplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional packages
from matplotlib.ticker import MaxNLocator

# LaTeX support
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

""" ------------------------------------------------------------------------------

    Main function

    >> Input: ECG signal
    >> Output: Plots and Statistics (in pandas format)

    >> Parameters:
       - at_time: instant at which the tracking starts
       - time_window: interval of time for inspection
       - patient: for labeling purposes only
       - showdata: if True, the discrete data correspondin to hear rate is displayed
         otherwise only the interpolation is plotted
       - bandwidth: default value is set to 2.0, it is a measure of smoothness for the
         interpotalion of the RR-distribution

------------------------------------------------------------------------------ """


def analysis(ecgsig,
             sampling_rate,
             at_time,
             time_window,
             patient,
             graphics=False,
             getpeaks=False,
             showdata=False,
             bandwidth=2.
             ):
    """-----------------------
       GROUP 1
    -----------------------"""

    # Time interval
    tau = np.arange(sampling_rate * at_time, sampling_rate * (at_time + time_window))

    # Generate the corresponding time domain
    tmp = np.arange(0, len(ecgsig) / sampling_rate, 1 / sampling_rate)

    # Transform to time format
    xtime = [time.strftime('%H:%M:%S', time.gmtime(tmp[tau][i])) for i in range(len(tmp[tau]))]

    # Time domain in points
    xpoint = np.arange(0, len(tau))

    # Truncate ECG signal
    ecg = ecgsig[tau]

    # Stamps to save plots
    stamps = str(at_time)

    """-----------------------
       GROUP 2
    -----------------------"""

    # Localisation of R-peaks
    rpeaks = findpeaks(ecg, sampling_rate)
    rpeaks = removeNAN(rpeaks)

    # Call peaks localisation algorithm
    waves = ecgwave(ecg, rpeaks, sampling_rate)

    # Extract ECG-peaks locations
    wpeaks = {}

    for feature, values in waves.items():
        wpeaks[feature] = [x for x in values if x > 0 or x is np.nan]

    ppeaks = wpeaks['pwaves']
    tpeaks = wpeaks['twaves']

    """-----------------------
       GROUP 3
    -----------------------"""

    # Compute R, P and T distributions
    RRI, rbinEdges, rweights, x_axis, rdist = ecgdists(rpeaks, sampling_rate, bandwidth)
    _, _, _, _, pdist = ecgdists(ppeaks, sampling_rate, bandwidth)
    _, _, _, _, tdist = ecgdists(tpeaks, sampling_rate, bandwidth)

    """-----------------------
       GROUP 4
    -----------------------"""
    # Compute statistics of R-R intervals
    period = np.ediff1d(rpeaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Actual heart rate
    hrpoints = 60 / period

    # -------------------------------------------------------/
    # Fitting heart rate data
    x0 = np.linspace(rpeaks[0], rpeaks[-1], len(hrpoints))
    x0 = x0[:, np.newaxis]
    y0 = hrpoints[:, np.newaxis]

    # Extract polynomial features from domain
    pf = PolynomialFeatures(degree=9)
    xp = pf.fit_transform(x0)

    # Select model
    model = LinearRegression()
    model.fit(xp, y0)

    # Output
    y1 = model.predict(xp)

    # New domain: to interpolate heart rate
    x_new = np.linspace(rpeaks[0], rpeaks[-1], len(ecg))

    # Once the array *period* has been computed,
    # an interpolation routine is invoked to plot smooth curves
    period = scipy.interpolate.PchipInterpolator(rpeaks, period, axis=0, extrapolate=None)

    # Compute actual periodicity of ECG signal
    period = 60 / period(x_new)

    # Mean heart rate
    rate_mean = period.mean()

    """-----------------------
       GROUP 5
    -----------------------"""
    if graphics:

        # Font size of axes labels and plot ticks
        fs = 18
        fl = fs - 4

        # Define figure
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 8))

        """-----------------------
           SUB-GROUP 5.1
        -----------------------"""

        # Plot interpolated heart rate and its mean
        if showdata:
            ax1.plot(np.linspace(rpeaks[0], rpeaks[-1], len(hrpoints)),
                     hrpoints,
                     color="#6BDE85",
                     linestyle='None',
                     marker='o',
                     markersize=10,
                     label="Heart Rate",
                     alpha=0.5, zorder=3)

        # Plot interpolation and fit
        ax1.plot(x0, y1, color="#3E4BD7", linewidth=2, label="Fit", zorder=4)
        ax1.plot(x_new, period, color="#5EA26C", linewidth=2, label="Interpolation", zorder=2)
        ax1.axhline(y=rate_mean, linestyle="--", color="#3B1B93", alpha=0.5, label="Mean", zorder=1)

        # Set title
        ax1.set_title('Instantaneous Heart Rate', fontsize=fs)

        # Configure plot ticks
        ax1.tick_params(axis='both', which='major', labelsize=fl)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_xticks(xpoint, labels=xtime, rotation=45)
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=8))

        # Plot labels
        ax1.set_xlabel(r'time', fontweight='bold', fontsize=fs)
        ax1.set_ylabel('Heart Rate', fontweight='bold', fontsize=fs)

        # Legends
        ax1.legend(loc='best')

        """-----------------------
           SUB-GROUP 5.2
        -----------------------"""

        # Histogram
        ax2.hist(rbinEdges[: -1], rbinEdges, weights=rweights, color="#684290", edgecolor="w", zorder=1, alpha=0.7)

        # Distribution
        ax2.plot(x_axis, rdist, color="#3B1B93", linewidth=2, zorder=2)

        # Data points
        ax2.scatter(RRI, np.full(len(RRI), np.max(rweights) / 10), c="k", alpha=0.3, marker="o", s=85, zorder=3)

        # Boxplot for percentiles
        ax2.boxplot(RRI,
                    vert=False,
                    patch_artist=True,
                    positions=[np.max(rweights) / 10],
                    widths=np.max(rweights) / 10,
                    manage_ticks=False,
                    boxprops=dict(linewidth=2),
                    medianprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                    zorder=4)

        # Set title
        ax2.set_title('Heart Rate Distribution', fontsize=fs)

        # Configure plot ticks
        ax2.tick_params(axis='both', which='major', labelsize=fl)

        # Axes labels
        ax2.set_xlabel('R-R intervals [sec]', fontweight='bold', fontsize=fs)
        ax2.set_ylabel('Occurrences', fontweight='bold', fontsize=fs)

        plt.savefig(patient + '_at_time_' + stamps + '.pdf', format='pdf')

        plt.close()

    """-----------------------
       SUB-GROUP 5.3
    -----------------------"""

    if getpeaks:
        # Define figure
        fig, axw = plt.subplots(constrained_layout=True, figsize=(8, 5))

        # Plot ECG signal and its wave peaks
        axw.plot(xpoint, ecg, linestyle='-', lw=2, color='black', alpha=0.5)
        axw.scatter(xpoint[ppeaks], ecg[ppeaks], color="#BE554B", s=200, alpha=0.7, label='P-peaks')
        axw.scatter(xpoint[rpeaks], ecg[rpeaks], color="#600967", s=200, alpha=0.7, label='R-peaks')
        axw.scatter(xpoint[tpeaks], ecg[tpeaks], color="#48C6E0", s=200, alpha=0.7, label='T-waves')

        # Plot ticks
        plt.xticks(xpoint, xtime, fontsize=14, rotation='45')
        plt.locator_params(axis='x', nbins=20)
        plt.yticks(fontsize=14)

        # Limit x-domain
        plt.xlim(min(xpoint), 0.1 * max(xpoint))

        # Plot labels
        plt.xlabel(r'time', fontweight='bold', fontsize=18)
        plt.ylabel(r'$V(t)$', fontsize=18)

        # Plot legends
        plt.legend(loc='best', fontsize=16)

        # Save plots
        plt.savefig('ECG_peaks_' + patient + '_at_time_' + stamps + '.pdf', format='pdf')

        # Show figures
        plt.show()

    # Final checks
    ppeaks = fitlen(rpeaks, ppeaks)
    tpeaks = fitlen(rpeaks, tpeaks)

    # Output
    return dict(R_t=rdist, P_t=pdist, T_t=tdist), dict(R=rpeaks, P=ppeaks, T=tpeaks), dict(HR=period)


#   end of main function --------------/

"""
    Calling main function 1.
    ------------------------
"""


def controlpanel(ecg,
                 sampling_rate,
                 start=0,
                 time_width=1,
                 n_iterations=None,
                 patient='00000',
                 graphics=False,
                 getpeaks=False,
                 showdata=False,
                 bandwidth=2.):
    # Time window, for statistics choose windows of 5, 15 and 30 min
    seconds = 60  # seconds
    numleap = int(time_width * seconds)

    # Define number of iterations
    if n_iterations is not None:
        iterations = n_iterations
    else:
        iterations = int((len(ecg) / sampling_rate - start) / numleap)

    # Checks [!]
    if not numleap * (iterations - 1) < int(len(ecg) / sampling_rate - start):
        raise ValueError("The time window and number of iterations must not exceed the length of signal")

    # Record data for statistics: DO NOT MODIFY
    for n in range(iterations):
        # Call main function
        dists, peaks, hr = analysis(ecg,
                                    sampling_rate,
                                    start + n * numleap,
                                    numleap,
                                    patient,
                                    graphics=graphics,
                                    getpeaks=getpeaks,
                                    showdata=showdata,
                                    bandwidth=bandwidth)

        # Save distributions and peaks
        disth5 = pd.DataFrame.from_dict(dists)
        peakh5 = pd.DataFrame.from_dict(peaks)
        rateh5 = pd.DataFrame.from_dict(hr)

        disth5.to_hdf(patient + '-t_width-' + str(time_width) + '-dists.h5', key='df' + str(n), mode='a')
        peakh5.to_hdf(patient + '-t_width-' + str(time_width) + '-peaks.h5', key='df' + str(n), mode='a')
        rateh5.to_hdf(patient + '-t_width-' + str(time_width) + '-hrate.h5', key='df' + str(n), mode='a')

        print('>> Status: ',
              (n + 1) * numleap,
              ' out of ', int(time_width * iterations),
              ' minutes of ECG signal analised')

    # Finally, verify and close hdf's
    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-dists.h5', 'r')
    datah5.close()

    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-peaks.h5', 'r')
    datah5.close()

    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-hrate.h5', 'r')
    datah5.close()

    print(">> Done!")


# End of body ---------------------------------------------------------/

"""
    Calling main function 2.
    ------------------------
"""


def cpanel2(ecg, sampling_rate,
            start=0,  # seconds
            time_width=60,  # seconds
            n_iterations=None,  # None by default to scan the whole time-series
            patient='00000',
            graphics=False,
            getpeaks=False,
            showdata=False,
            bandwidth=2.):
    # Time window, for statistics choose windows of 5, 15 and 30 min
    overlap = 10  # seconds
    numleap = int(time_width)

    # Define number of iterations
    if n_iterations is not None:
        iterations = n_iterations
    else:
        iterations = int((len(ecg) / sampling_rate - start) / numleap)

    # Checks [!]
    if not numleap * (iterations - 1) < int(len(ecg) / sampling_rate - start):
        raise ValueError("The time window and number of iterations must not exceed the length of signal")

    # Record data for statistics: DO NOT MODIFY
    for n in range(iterations):
        # Call main function
        dists, peaks, hr = analysis(ecg,
                                    sampling_rate,
                                    start + n * (numleap - overlap),
                                    numleap,
                                    patient,
                                    graphics=graphics,
                                    getpeaks=getpeaks,
                                    showdata=showdata,
                                    bandwidth=bandwidth)

        # Save distributions and peaks
        disth5 = pd.DataFrame.from_dict(dists)
        peakh5 = pd.DataFrame.from_dict(peaks)
        rateh5 = pd.DataFrame.from_dict(hr)

        disth5.to_hdf(patient + '-t_width-' + str(time_width) + '-dists.h5', key='df' + str(n), mode='a')
        peakh5.to_hdf(patient + '-t_width-' + str(time_width) + '-peaks.h5', key='df' + str(n), mode='a')
        rateh5.to_hdf(patient + '-t_width-' + str(time_width) + '-hrate.h5', key='df' + str(n), mode='a')

        print('>> Status: ',
              n * numleap,
              ' out of ', int(time_width * iterations),
              ' seconds of ECG signal of patient ', patient)

    # Verify and close hdf's
    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-dists.h5', 'r')
    datah5.close()

    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-peaks.h5', 'r')
    datah5.close()

    datah5 = h5py.File(patient + '-t_width-' + str(time_width) + '-hrate.h5', 'r')
    datah5.close()

    print(">> Done!")


# End of body ---------------------------------------------------------/

"""
    Preliminar filtering of ECG signals.
    ------------------------------------
"""


def preFilter(lowcut=None, highcut=None, sampling_rate=125, normalize=False):
    # Checking Nyquist frequency
    if isinstance(highcut, int):
        if sampling_rate <= 2 * highcut:
            warn("The sampling rate is below Nyquist frequency")

    # Replace 0 by none
    if lowcut is not None and lowcut == 0:
        lowcut = None
    if highcut is not None and highcut == 0:
        highcut = None

    # Classification
    freqs = 0
    filter_type = ""
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filter_type = "bandstop"
        else:
            filter_type = "bandpass"
        freqs = [lowcut, highcut]

    elif lowcut is not None:
        freqs = [lowcut]
        filter_type = "highpass"

    elif highcut is not None:
        freqs = [highcut]
        filter_type = "lowpass"

    # Offset: fixing frequency to Nyquist frequency
    if normalize is True:
        freqs = np.array(freqs) / (sampling_rate / 2)

    return freqs, filter_type


"""
    Butterworth filter for ECG signals.
    -----------------------------------
"""


def filterECG(signal, sampling_rate):
    # Apply high-pass, Butterworth filter to ECG signal
    freqs, filter_type = preFilter(lowcut=0.5, highcut=None, sampling_rate=sampling_rate)

    # Compute sos via scipy
    order = 5
    sos = scipy.signal.butter(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
    filtered = scipy.signal.sosfiltfilt(sos, signal)

    # Apply filter to remove dc
    powerline = 50

    # Preparing all arrays
    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]

    return scipy.signal.filtfilt(b, a, filtered, method="pad")


"""
    Compute distributions and histograms.
    -------------------------------------
"""


def ecgdists(xpeaks, sampling_rate, bandwidth=2.):
    # Compute differences between R-peaks to obtain their distribution over time
    XXI = np.diff(xpeaks) / sampling_rate

    # Range for uniform bins
    rngb = np.arange(0.5, 40) / 20

    # Compute the numpy histogram of X-peaks (R, P or T peaks)
    y, binEdges = np.histogram(XXI, bins=rngb, density=True, normed=True)
    bincenters = 0.5 * (binEdges[1:] + binEdges[: -1])

    # Compute a density function to fit data
    rho = scipy.stats.gaussian_kde(XXI, bw_method="silverman")
    rho.set_bandwidth(bw_method=rho.factor / bandwidth)

    # Number of points for interpolation
    NPOINTS = 500

    # New domain and interpolated range
    x_axis = np.linspace(np.min(rngb), np.max(rngb), num=NPOINTS)

    # Re-scaling the interpolated distribution
    if rho(x_axis) is list:
        y_axis = list(rescale(np.array(rho(x_axis)), to=[0, max(y)], scale=None))
    else:
        y_axis = rescale(rho(x_axis), to=[0, max(y)], scale=None)

    return XXI, binEdges, y, x_axis, y_axis


"""
    Locate R-peaks in ECG signals.
    ------------------------------
"""


def findpeaks(
        signal,
        sampling_rate=125,
        smoothwindow=0.1,
        avgwindow=0.75,
        gradthreshweight=1.5,
        minlenweight=0.4,
        mindelay=0.3
):
    # Use numpy.gradient() to locate ECG threshold
    grad = np.gradient(signal)

    # For computational purposes compute the absolute value of the gradient
    absgrad = np.abs(grad)

    # Fixing parameters
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))

    # Average parameter to fix kernel
    avg_kernel = int(np.rint(avgwindow * sampling_rate))

    # Smooth ECG signals for locating complex points
    smoothgrad = scipy.ndimage.uniform_filter1d(absgrad, size=smooth_kernel, mode="nearest")
    avggrad = scipy.ndimage.uniform_filter1d(smoothgrad, size=avg_kernel, mode="nearest")

    gradthreshold = gradthreshweight * avggrad

    # Recalculate min-delay
    mindelay = int(np.rint(sampling_rate * mindelay))

    # Locate QRS complexes
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0: -1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0: -1], np.logical_not(qrs[1:])))[0]

    # Neglect R-peaks that precede first QRS-start
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    # Neglect R-peaks that are too short
    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    peaks = [0]

    # Inspect all start and end points of QRS-complexes
    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:

            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]

            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    # Prepare outcome
    peaks.pop(0)
    peaks = np.asarray(peaks).astype(int)

    """
        Further improvements: Correction in the location of R-peaks according to the reference:

        Jukka A. Lipponen & Mika P. Tarvainen (2019):
        A robust algorithm for heart rate variability time series artefact correction using novel beat
        classification, Journal of Medical Engineering & Technology, DOI: 10.1080/03091902.2019.1640306.

    """
    return peaks


"""
    Scanning ECG waves.
    -------------------
"""


def ecgwave(ecg, rpeaks, sampling_rate=200, new_rate=2000):
    # ECG signal must be artificially resampled
    # 1. Auxiliar lengths
    auxlen = int(np.round(len(ecg) * new_rate / sampling_rate))

    # 2. Interpolation stage
    ecg = scipy.ndimage.zoom(ecg, auxlen / len(ecg))

    # Wavelet transforms (multiscale)
    dwtmatr = wavelets(ecg, 10)

    rpeaks_resampled = points(rpeaks, sampling_rate, new_rate)

    tpeaks, ppeaks = scanPTWaves(ecg, rpeaks_resampled, dwtmatr, sampling_rate=new_rate)

    return dict(
        pwaves=points(ppeaks, new_rate, aux_rate=sampling_rate),
        twaves=points(tpeaks, new_rate, aux_rate=sampling_rate)
    )


""" ------------------------------------------------------------
    >>> SUPPORT FUNCTIONS [DO NOT MODIFY!]
    ------------------------------------------------------------
"""


def scanPTWaves(
        ecg,
        rpeaks,
        dwtmatr,
        sampling_rate=250,
        qrs_width=0.13,
        p2r_duration=0.2,
        rt_duration=0.25,
        degree_tpeak=3,
        degree_ppeak=2,
        epsilon_T_weight=0.25,
        epsilon_P_weight=0.02,
):
    """
    This function is only temporary
    The new algorithm will be in scanECGwave()
    Future function will be called directly from ecgwave()
    """
    srch_bndry = int(0.5 * qrs_width * sampling_rate)
    degree_add = dwtparams(rpeaks, sampling_rate, target='degree')

    # Fitting parameters for DWT method
    p2r_duration = dwtparams(rpeaks, sampling_rate, duration=p2r_duration, target='duration')
    rt_duration = dwtparams(rpeaks, sampling_rate, duration=rt_duration, target='duration')

    # Looking for T-waves ---------------------------/
    tpeaks = []
    for rpeak_ in rpeaks:

        # Checks !
        if np.isnan(rpeak_):
            tpeaks.append(0)
            continue

        # Search for T waves w.r.t. R waves
        srch_idx_start = rpeak_ + srch_bndry
        srch_idx_end = rpeak_ + 2 * int(rt_duration * sampling_rate)
        dwt_local = dwtmatr[degree_tpeak + degree_add, srch_idx_start:srch_idx_end]

        if len(dwt_local) == 0:
            tpeaks.append(0)
            continue

        height = epsilon_T_weight * np.sqrt(np.mean(np.square(dwt_local)))

        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))
        if dwt_local[0] > 0:
            peaks = [0] + peaks

        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
            if correct_sign:
                idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt + 1])[0] + idx_peak

                score = ecg_local[idx_zero] - (float(idx_zero) / sampling_rate -
                                               (rt_duration - 0.5 * qrs_width))
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            tpeaks.append(0)
            continue

        tpeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

    # Looking for P-waves ---------------------------/
    ppeaks = []
    for rpeak in rpeaks:

        # Checks !
        if np.isnan(rpeak):
            ppeaks.append(0)
            continue

        srch_idx_start = rpeak - 2 * int(p2r_duration * sampling_rate)
        srch_idx_end = rpeak - srch_bndry
        dwt_local = dwtmatr[degree_ppeak + degree_add, srch_idx_start:srch_idx_end]

        if len(dwt_local) == 0:
            ppeaks.append(0)
            continue

        height = epsilon_P_weight * np.sqrt(np.nanmean(np.square(dwt_local)))

        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))
        if dwt_local[0] > 0:
            peaks = [0] + peaks

        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0
            if correct_sign:
                idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt + 1])[0] + idx_peak

                score = ecg_local[idx_zero] - abs(
                    float(idx_zero) / sampling_rate - p2r_duration
                )
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            ppeaks.append(0)
            continue

        ppeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

    return tpeaks, ppeaks


"""
    Parameters for DWT method.
    ---------------------------
"""


def dwtparams(rpeaks, sampling_rate, duration=None, target=None):
    period = np.ediff1d(rpeaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Actual heart rate
    discrete_hr = 60 / period

    average_rate = np.median(discrete_hr)

    if target == "degree":
        scale_factor = (sampling_rate / 250) / (average_rate / 60)
        return int(np.log2(scale_factor))
    elif target == "duration":
        return np.round(duration * (60 / average_rate), 3)


"""
    Wavelet method.
    ---------------
"""


def wavelets(ecg, max_degree):
    def hfilter(signal_i, power=0):
        zeros = np.zeros(2 ** power - 1)
        timedelay = 2 ** power
        banks = np.r_[
            1.0 / 8, zeros, 3.0 / 8, zeros, 3.0 / 8, zeros, 1.0 / 8,
        ]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]
        return signal_f

    def gfilter(signal_i, power=0):
        zeros = np.zeros(2 ** power - 1)
        timedelay = 2 ** power
        banks = np.r_[2, zeros, -2]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]
        return signal_f

    dwtmatr = []
    intermediate_ret = np.array(ecg)

    for deg in range(max_degree):
        S_deg = gfilter(intermediate_ret, power=deg)
        T_deg = hfilter(intermediate_ret, power=deg)
        dwtmatr.append(S_deg)
        intermediate_ret = np.array(T_deg)

    dwtmatr = [arr[: len(ecg)] for arr in dwtmatr]

    return np.array(dwtmatr)


"""
    Detect zero-crossings.
    ----------------------
"""


def signal_zerocrossings(signal, direction="both"):
    df = np.diff(np.sign(signal))

    if direction in ["positive", "up"]:
        zerocrossings = np.where(df > 0)[0]
    elif direction in ["negative", "down"]:
        zerocrossings = np.where(df < 0)[0]
    else:
        zerocrossings = np.nonzero(np.abs(df) > 0)[0]

    return zerocrossings


"""
    Identify ECG points.
    --------------------
"""


def points(peaks, sampling_rate, aux_rate):
    if isinstance(peaks, np.ndarray):
        peaks = peaks.astype(dtype=np.int64)
    elif isinstance(peaks, list):
        peaks = np.array(peaks)

    newpeaks = peaks * aux_rate / sampling_rate
    newpeaks = [np.nan if np.isnan(x) else int(x) for x in newpeaks.tolist()]

    return newpeaks


"""
    Remove NANs from ECG peak-arrays.
    ---------------------------------
"""


def removeNAN(A):
    A = np.asanyarray(A)
    nonA = ~np.isnan(A)
    return A[nonA].astype(int)


"""
    Add NANs to fit ECG peak-arrays.
    --------------------------------
"""


def fitlen(a, b):
    if len(a) > len(b):
        for i in range(len(a) - len(b)):
            b = np.append(b, np.nan)
    else:
        b = b
    return b


"""
    Re-scaling of pseudo-Gaussian distributions.
    --------------------------------------------
"""


def rescale(data, to=None, scale=None):
    if to is None:
        to = [0, 1]
    if scale is None:
        scale = [np.nanmin(data), np.nanmax(data)]
    return (to[1] - to[0]) / (scale[1] - scale[0]) * (data - scale[0]) + to[0]


"""
    Polynomial for fitting heart rate data.
    ---------------------------------------
"""


def func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5 + a6 * x ** 6 + a7 * x ** 7 + a8 * x ** 8


"""
    Additional plots for n-distributions.
    -------------------------------------
"""


def distcollection(distribution, xdom, title='Distribution of peaks'):
    # Support libraries
    from matplotlib import cm

    # Ticks for x-axis
    xtks = (60 / xdom).astype(int)

    # Close all plots
    plt.close()

    # Define mesh
    X, Y = np.meshgrid(xdom, np.arange(distribution.shape[0]))

    # Set color map
    norm = plt.Normalize(Y.min(), Y.max())
    colors = cm.Dark2(norm(Y))

    # Define figure and subplot
    fig = plt.figure(constrained_layout=True, figsize=(11, 8))
    ax3 = fig.add_subplot(111, projection='3d')

    # Surf plot
    surf = ax3.plot_surface(X, Y, distribution,
                            rcount=20,
                            ccount=1,
                            lw=2,
                            facecolors=colors,
                            shade=False)

    # Point of view
    ax3.view_init(20, -120)

    # Remove shade and color between wires
    surf.set_facecolors([0, 0, 0, 0])

    # Fontsize
    fs = 18

    # Axes labels and ticks
    ax3.set_xlabel(title + ' [sec]', fontsize=fs)
    ax3.set_ylabel('Bins', fontsize=fs, rotation=-40)
    ax3.set_zlabel('Average events', fontsize=fs)

    ax3.xaxis.set_tick_params(labelsize=0.7 * fs)
    ax3.yaxis.set_tick_params(labelsize=0.7 * fs)
    ax3.zaxis.set_tick_params(labelsize=0.7 * fs)

    # Remove grid and panes
    # ax3.grid(False)
    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    fig.suptitle('Assortment of ' + title, fontweight='bold', fontsize=1.2 * fs)

    # Show plot
    plt.show()


"""
    Statistical mechanical analysis of the nth distribution.
    --------------------------------------------------------
"""
import seaborn as sns
from matplotlib.patches import Ellipse


def diagrams(domain, peaks, distribution, sample_rate, nbin, name):
    # Compute statistical velocity and acceleration
    dens = distribution[nbin]
    velo = np.diff(dens)
    acce = np.diff(velo)

    velo = np.append(velo, [0])
    acce = np.append(acce, [0, 0])

    # Compute differences for Poincare plot
    peaks = removeNAN(peaks[nbin])
    x = np.ediff1d(peaks, to_begin=0) / sample_rate
    xn, xn1 = x[: -1], x[1:]

    sd1 = np.sqrt(0.5) * np.std(xn1 - xn)
    sd2 = np.sqrt(0.5) * np.std(xn1 + xn)

    # ----------------------------------------------------------------/
    # Clean all plots
    plt.close()

    # Define new figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(11, 8))

    # Setup grid
    ax1 = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=5)
    ax2 = plt.subplot2grid((10, 10), (5, 0), colspan=5, rowspan=5)
    ax3 = plt.subplot2grid((10, 10), (5, 5), colspan=6, rowspan=5)
    ax4 = plt.subplot2grid((10, 10), (0, 5), colspan=5, rowspan=5)

    # Fontsize
    fs = 14
    lw = 3

    # ----------------------------------------------------------/
    # Distributions
    ax1.plot(domain, dens, lw=lw, color='#963FB3', alpha=0.7)

    ax1.set_xlabel('peak to peak intervals [s]', fontsize=fs)
    ax1.set_ylabel('average events', fontsize=fs)

    ax1.tick_params(axis='both', which='major', labelsize=0.8 * fs)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))
    # ----------------------------------------------------------/
    # Velocity and acceleration
    ax2.plot(domain, velo, '-', lw=lw, color='#3D7BA5', label='Velocity')
    ax2.plot(domain, acce, '--', lw=lw, color='#3ECF7D', label='Acceleration')

    ax2.set_xlabel('peak to peak intervals [s]', fontsize=fs)
    ax2.set_ylabel('rate', fontsize=fs)

    ax2.legend(loc='best', fontsize=0.8 * fs)
    # ----------------------------------------------------------/
    # Phase portrait
    ax3.plot(velo, acce, '-o', color='#E87404', lw=lw, ms=6, alpha=0.5)

    ax3.set_xlabel('velocity', fontsize=fs)
    ax3.set_ylabel('acceleration', fontsize=fs)
    # ----------------------------------------------------------/
    # Poincare plot
    sns.scatterplot(x=xn, y=xn1, color='#38B567', alpha=0.5)
    ellipse = Ellipse((np.mean(x), np.mean(x)),
                      2 * sd1,
                      2 * sd2,
                      lw=2,
                      fill=False,
                      color='#252076',
                      alpha=0.5,
                      angle=-45)
    ax4.add_patch(ellipse)

    ax4.arrow(np.mean(x),
              np.mean(x),
              0.5 * (np.max(x) - np.min(x)),
              0.5 * (np.max(x) - np.min(x)),
              color='grey',
              linewidth=lw, alpha=0.3)

    ax4.arrow(np.mean(x),
              np.mean(x),
              -sd1 * np.sqrt(0.5),
              sd1 * np.sqrt(0.5),
              color='#6A41A5',
              linewidth=lw, label='SD1')

    ax4.arrow(np.mean(x),
              np.mean(x),
              sd2 * np.sqrt(0.5),
              sd2 * np.sqrt(0.5),
              color='#2578F0',
              linewidth=lw, label='SD2')

    ax4.set_xlabel(r'$xx_n$ [s]', fontsize=fs)
    ax4.set_ylabel(r'$xx_{n+1}$ [s]', fontsize=fs)
    ax4.legend(loc='best', fontsize=0.8 * fs)
    # ----------------------------------------------------------/
    # Set title
    fig.suptitle('Statistical dynamics of ' + name + ' at bin ' + str(nbin), fontweight='bold', fontsize=1.1 * fs)

    # Display plot
    plt.show()


""" ---------------------------------------
    E O F
----------------------------------------"""
