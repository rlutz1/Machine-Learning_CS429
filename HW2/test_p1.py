import numpy as np
from P1 import LinearSVC

# Simple toy dataset (linearly separable)
X = np.array([
    [2, 3],
    [1, 1],
    [2, 1],
    [3, 2],
    [-1, -1],
    [-2, -1],
    [-3, -2]
])

# Labels must be {-1, +1}
y = np.array([1, 1, 1, 1, -1, -1, -1])

# Create the model
svc = LinearSVC(eta=0.01, epochs=100, C=1.0)

# Train the model
svc.fit(X, y)

# Make the predictions pls
predictions = svc.predict(X)

print("Predictions:", predictions)
print("Weights:", svc.w_)
print("Bias:", svc.b_)
