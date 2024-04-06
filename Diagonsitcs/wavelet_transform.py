import matplotlib.pyplot as plt
import pandas as pd
import pywt

# Load the dataset from a CSV file
csv_path = input("Enter csv path: ")
data = pd.read_csv(csv_path)

signal = data['218'].values

# Perform a single-level Discrete Wavelet Transform
coeffs = pywt.dwt(signal, 'db4')
cA, cD = coeffs

# cA are the approximation coefficients
# cD are the detail coefficients

# Plotting the original signal and the transformed components
plt.figure(figsize=(12, 8))

# Original signal
plt.subplot(311)
plt.plot(data['time'], signal, label='Original Signal')
plt.legend()

# Approximation coefficients
plt.subplot(312)
plt.plot(cA, label='Approximation Coefficients')
plt.legend()

# Detail coefficients
plt.subplot(313)
plt.plot(cD, label='Detail Coefficients')
plt.legend()

plt.tight_layout()
plt.show()
