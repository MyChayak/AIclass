import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd
from sklearn.datasets import make_blobs

# Generate synthetic data using make_blobs
X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2.0, 2.0),
                    cluster_std=0.75,
                    random_state=69)

X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3.0, 3.0),
                    cluster_std=0.75,
                    random_state=69)

y2 = np.ones_like(y2) 

# Combine the two datasets
X = np.concatenate([X1, X2], axis=0)
y = np.concatenate([y1, y2], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

# Build and compile the model
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(0.01)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=200, verbose=0)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob).astype(int).ravel()

# Decision boundary plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

# Plot the decision boundary and points
plt.contourf(xx, yy, Z, alpha=0.4)  # Contour plot for decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolors='k')  # Scatter plot with colors
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
