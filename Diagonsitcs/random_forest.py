import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(input("Enter csv path:"))

target_column_names = ['max_wait_time', 'min_wait_time', 'queue_length', 'total_cars']
X = df.drop(target_column_names, axis=1)  # Features
y = df[target_column_names]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = [f"{i}" for i in X.columns]
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
