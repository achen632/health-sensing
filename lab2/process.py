# Scientific packages
import numpy as np
from numpy import linspace, diff, zeros_like, arange, array
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, spectrogram, butter, filtfilt, find_peaks
from scipy.fft import  rfft, rfftfreq

## Apply moving window averaging & Savitzky–Golay filter
def moving_average(signal, window_size):
    cumsum = np.cumsum(np.insert(signal, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def lowpass(signal, fs, cutoff):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def highpass(signal, fs, cutoff):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def process(filepath, height, distance, moving, savgol, poly, cutoff, high_cutoff=0.1):
    # Sampling frequency of acquired data
    fs = 200 #200Hz

    # TODO: load ECG signal using genfromtxt()
    signal = np.genfromtxt(filepath)[5000:-3000]

    length = len(signal)
    period = 1/fs
    total_time = length * period
    time = np.arange(0, total_time, period)
    if len(time) > length:
        time = time[:length]

    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label='ECG Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Raw Data')
    plt.title('ECG Acquisition')
    plt.legend()
    plt.grid(True)
    plt.show()

    mov_avg_sig = moving_average(signal, window_size=moving)

    time = time[len(time) - len(mov_avg_sig):]

    plt.figure(figsize=(12, 6))
    plt.plot(time, mov_avg_sig, label='moving window avg Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Raw Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    savgol_signal = savgol_filter(mov_avg_sig, window_length=savgol, polyorder=poly)

    plt.figure(figsize=(12, 6))
    plt.plot(time, savgol_signal, label='sav gol filtered Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Raw Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    # TODO: Implement the low-pass filter using lowpass() in biosignalsnotebook.process
    filtered_signal = lowpass(savgol_signal, fs, cutoff=cutoff)
    filtered_signal = highpass(filtered_signal, fs, cutoff=high_cutoff)

    # clip first 5 seconds of data
    time = time[1000:]
    filtered_signal = filtered_signal[1000:]

    plt.figure(figsize=(12, 6))
    plt.plot(time, filtered_signal, label='Post Filtering ECG Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Raw Data')
    plt.title('ECG Acquisition')
    plt.legend()
    plt.xlim(left = 20)
    plt.show()

    # TODO: Normalize data with mean & std
    norm_signal = filtered_signal - np.mean(filtered_signal)
    norm_signal = norm_signal / np.std(norm_signal)

    plt.figure(figsize=(12, 6))
    plt.plot(time, norm_signal, label='Normalized ECG Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Data')
    plt.title('ECG Acquisition')
    plt.legend()
    plt.grid(True)
    plt.show()

    np_norm_signal = np.array(norm_signal)

    f, t, Sxx = spectrogram(np_norm_signal, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim([0,2])
    plt.show()

    # TODO: Compute the Fourier Transform of the filtered signal using rfft() from scipy.fft
    fft_vals = rfft(norm_signal)

    # TODO: Compute all frequencies present in the signal using rfftfreq() from scipy.fft
    fft_freq = rfftfreq(len(norm_signal), 1/fs)

    peaks = find_peaks(np.abs(fft_vals), height=height, distance=distance)[0]

    plt.figure(figsize=(12, 6))
    plt.plot(fft_freq, np.abs(fft_vals), label='Frequency Spectrum')
    plt.plot(fft_freq[peaks], np.abs(fft_vals)[peaks], 'o', label='Peaks')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum of Normalized ECG Signal')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

    dominant_frequency = fft_freq[peaks[0]]
    print(f"Dominant Frequency: {dominant_frequency:.4f} Hz")
    print(f"Breath Rate: {dominant_frequency * 60:.2f} breaths per minute")