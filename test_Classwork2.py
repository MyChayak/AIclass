import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def weighted_sum(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0,1,-1)
    
    def fit(self,X,y):
        #initializing the weights to 0
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        print("Weights:", self.w_)
        
        #training the model n_iter times
        for _ in range(self.n_iter):
            error = 0

            #loop through each input
            for xi, y in zip(X,y):
                y_pred = self.predict(xi)               # Calculate ŷ (predicted value)
                update = self.eta*(y-y_pred)            # Calculate Update
                self.w_[1:] = self.w_[1:] + update*xi   # Update the weights
                print("Update Weights:", self.w_[1:])   
                self.w_[0] = self.w_[0]+update          # Update the bias (X0 = 1)
                error += int(update != 0.0)             # if update != 0, then ŷ != y
            self.errors_.append(error)
        return self


# Generation Data Sample
X1,y1 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(2.0,2.0),
                   cluster_std=0.25,
                   random_state=69)

X2,y2 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(3.0,3.0),
                   cluster_std=0.25,
                   random_state=69)

print(y1)
print(y2)
# Plot decision plane
def decision_function(x1,x2):
    return x1+x2 - 0.5

# Generate grid of points
x1_range = np.linspace(-1,2,500)
x2_range = np.linspace(-1,2,500)
x1_grid, x2_grid = np.meshgrid(x1_range,x2_range)

# Evalue the desicion function on the grid
g_values = decision_function(x1_grid, x2_grid)

# Plot dataset
fig = plt.figure()
fig.suptitle("Data Sample")
plt.scatter(X1[:,0], X1[:,1], c='red', linewidths = 1, alpha = 0.6, label = "Class 1")
plt.scatter(X2[:,0], X2[:,1], c='blue', linewidths = 1, alpha = 0.6, label = "Class 2")
plt.xlabel('Feature 1', fontsize=10)
plt.ylabel('Feature 2', fontsize=10)
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()
fig.savefig('Out1 - Data_Sample.png')

# Plot the decision boundary and the decision regions
plt.figure()
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf,0,np.inf],
             colors=['red','blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black',
                         linewidths=2)
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Desicion Plane')
plt.grid(True)
plt.show()