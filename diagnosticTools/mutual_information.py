import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

# Load the dataset
csv_path = input("Enter csv path: ")
df = pd.read_csv(csv_path)

# Assuming 'tls_id' needs one-hot encoding
df = pd.get_dummies(df, columns=['tls_id'], prefix='tls_id')

target_column_name = 'queue_length'  # Target variable
X = df.drop(target_column_name, axis=1)  # Drop all target variables to isolate features
y = df[target_column_name]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate Mutual Information
mi_scores = mutual_info_regression(X_train, y_train)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)

# Sort scores
mi_scores = mi_scores.sort_values(ascending=False)

# Visualizing Mutual Information scores
mi_scores.plot.bar()
plt.title('Mutual Information scores', fontsize=20)
plt.ylabel('MI Scores', fontsize=16)
plt.xticks(rotation=45, fontsize=6)  # Rotate the labels for better readability
plt.show()
