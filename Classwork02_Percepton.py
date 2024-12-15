import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        return np.array([self.step_function(self.weighted_sum(xi)) for xi in X])
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                y_pred = self.predict([xi])[0]
                update = self.eta * (target - y_pred)
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

# Generate Data
X1, y1 = make_blobs(n_samples=100, n_features=2, centers=1, center_box=(2.0, 2.0), cluster_std=0.25, random_state=69)
X2, y2 = make_blobs(n_samples=100, n_features=2, centers=1, center_box=(3.0, 3.0), cluster_std=0.25, random_state=69)

# Combine the data
X = np.vstack((X1, X2))
y = np.hstack((np.ones(y1.shape[0]), np.zeros(y2.shape[0])))

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Train Perceptron
perc = Perceptron(eta=0.1, n_iter=10)
perc.fit(X_std, y)

# Define decision function similar to the first code
def decision_function(x1, x2):
    return perc.w_[0] + perc.w_[1] * x1 + perc.w_[2] * x2

# Create a grid to visualize the decision boundary
x1_range = np.linspace(-1.5, 2, 500)
x2_range = np.linspace(-1.5, 2, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

g_values = decision_function(x1_grid, x2_grid)

def plot_decision_boundary(X, y, weights):
    plt.figure()
    # Plot the points for Class 1 (red) and Class 2 (blue)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 2')
    
    # Create a grid to plot the decision boundary
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]  # Decision Boundary formula: w0 + w1*x1 + w2*x2 = 0
    
    plt.plot(x_vals, y_vals, color='black', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()

# Call the plot_decision_boundary function
plot_decision_boundary(X_std, y, perc.w_)

# Plot dataset
fig = plt.figure()
fig.suptitle("Data Sample")
plt.scatter(X1[:, 0], X1[:, 1], c='red', linewidths=1, alpha=0.6, label="Class 1")
plt.scatter(X2[:, 0], X2[:, 1], c='blue', linewidths=1, alpha=0.6, label="Class 2")
plt.xlabel('Feature 1', fontsize=10)
plt.ylabel('Feature 2', fontsize=10)
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()

# Plot decision boundary
plt.figure()
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Boundary')
plt.grid(True)
plt.show()
