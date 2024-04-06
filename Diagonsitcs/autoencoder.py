import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
csv_path = input("Enter csv path: ")
df = pd.read_csv(csv_path)

# Assuming 'tls_id' needs one-hot encoding
df = pd.get_dummies(df, columns=['tls_id'], prefix='tls_id')

# Preprocess the data: scale features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# scaled_data is the preprocessed data
input_dim = scaled_data.shape[1]
encoding_dim = 8  # Based on PCA results

# Define the input layer
input_layer = Input(shape=(input_dim,))
# Define the encoding layers
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# Define the decoding layers
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)
# Define the encoder model
encoder = Model(input_layer, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(scaled_data, scaled_data,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)

# Use encoder to extract features
encoded_features = encoder.predict(scaled_data)
print(encoded_features)

# Calculate reconstruction
reconstructed_data = autoencoder.predict(scaled_data)

# Compute mean sqaured error for all features for each sample
mse = np.mean(np.power(scaled_data - reconstructed_data, 2), axis=1)

# make graph
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50)
plt.xlabel('Mean Squared Error')
plt.ylabel('Number of Samples')
plt.title('Distribution of Reconstruction Errors')
plt.show()
