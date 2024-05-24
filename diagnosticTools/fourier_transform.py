import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset from a CSV file
csv_path = input("Enter csv path: ")
df = pd.read_csv(csv_path)

# Extract time and signal values
t = df['time'].values
signal = df['218'].values

# Sampling details
fs = 1 / (t[1] - t[0])  # Calculate sampling frequency based on time difference
L = len(signal)  # Length of the signal

# Apply FFT to the signal
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(L, 1 / fs)[:L // 2]  # Get the frequency axis

# Calculate the magnitude of the FFT results
magnitude = np.abs(fft_result)[:L // 2] * 2 / L

# Plotting the original signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')

# Plotting the FFT results
plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude)
plt.title('Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

bands = {
    'f1': (0, 0.02),
    'f2': (0.02, 0.03),
    'f3': (0.03, 0.04),
    'f4': (0.04, 0.06),
    'f5': (0.06, 0.07),
    'f6': (0.07, 0.08),
    'f7': (0.08, 0.1)
}


def extract_brain_wave_band_power(fft_result, frequencies):
    global bands
    band_powers = {}
    for band, (low_freq, high_freq) in bands.items():
        # Find indices where frequencies fall within the current band
        indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))
        # Sum power (magnitude squared) of FFT components within the band
        band_power = np.sum(np.abs(fft_result[indices]) ** 2)
        band_powers[band] = band_power
    return band_powers


def calculate_band_powers(row):
    signal = row
    sampling_rate = fs

    # Mean correction
    signal -= np.mean(signal)

    # Perform FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1 / sampling_rate)

    # Extract positive frequencies only for analysis
    n = len(signal)
    fft_result_positive = fft_result[:n // 2]
    frequencies_positive = frequencies[:n // 2]

    # Extract brain wave band powers
    band_powers = extract_brain_wave_band_power(fft_result_positive, frequencies_positive)

    return band_powers


signal = pd.DataFrame(df['218'])
band_powers = calculate_band_powers(signal.values)
signal['band_powers'] = band_powers
# Extracting each band from 'band_powers' into its own column
for band in bands:
    signal[band] = band_powers[band]

plt.figure(figsize=(12, 8))
sns.boxplot(data=signal[bands.keys()])
plt.title('Distribution of Power in feature')
plt.xlabel('Feature')
plt.ylabel('Power')
plt.show()
