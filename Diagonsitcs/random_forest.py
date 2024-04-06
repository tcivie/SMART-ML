import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
csv_path = input("Enter csv path: ")
df = pd.read_csv(csv_path)

# Assuming 'tls_id' needs one-hot encoding
df = pd.get_dummies(df, columns=['tls_id'], prefix='tls_id')

target_column_name = 'queue_length'  # Target variable
X = df.drop(target_column_name, axis=1)  # Drop all target variables to isolate features
y = df[target_column_name]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Calculating feature importances
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
feature_names = X.columns

# Plotting feature importances
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()
