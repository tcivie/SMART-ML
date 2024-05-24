import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
csv_path = input("Enter csv path: ")
df = pd.read_csv(csv_path)

# Assuming 'tls_id' needs one-hot encoding
df = pd.get_dummies(df, columns=['tls_id'], prefix='tls_id')

target_column_name = 'queue_length'  # Target variable
X = df.drop(target_column_name, axis=1)  # Drop all target variables to isolate features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=None)  # None: keep all components, or set a specific number
X_pca = pca.fit_transform(X_scaled)

# Display the Explained Variance Ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o',
         linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
