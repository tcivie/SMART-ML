import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset from a CSV file
csv_path = input("Enter csv path: ")
data = pd.read_csv(csv_path)

signal = data['218'].values

# Activity
Activity = np.var(signal)

# Mobility
first_derivative = np.diff(signal)
mobility = np.sqrt(np.var(first_derivative) / Activity)

# Complexity
second_derivative = np.diff(first_derivative)
complexity = (np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / mobility)

# Calculated Hjorth Parameters
hjorth_params = {'Activity': Activity, 'Mobility': mobility, 'Complexity': complexity}

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(hjorth_params.keys(), hjorth_params.values(), color=['blue', 'orange', 'green'])
plt.title('Hjorth Parameters of the Signal')
plt.ylabel('Value')
plt.xlabel('Parameter')
plt.show()
