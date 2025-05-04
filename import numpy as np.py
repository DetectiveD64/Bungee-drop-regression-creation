import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fit_polynomial_regression(X, y, degree=3):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly

def polynomial_equation(model, poly, precision=6):
    coefs = model.coef_
    intercept = model.intercept_
    feature_names = poly.get_feature_names_out(['x', 'y'])
    terms = [f"{coefs[i]:.{precision}g} * {feature_names[i]}" for i in range(len(coefs))]
    equation = " + ".join(terms)
    return f"z = {intercept:.{precision}g} + {equation}"

# Given data
X = np.array([
    [220.5, 94.5], [140.5, 166.4], [151.5, 257.9], [140.5, 257.9], [90.5, 102.1],
    [185.5, 187.5], [233.5, 53.4], [121.5, 53.3], [121.5, 53.3], [121.5, 53.3],
    [171.5, 53.3], [171.5, 53.3], [90.5, 53.3], [130.5, 341.2],
    [130.5, 341.2], [90.5, 341.2], [90.5, 199], [161.5, 149], [151.5, 115.8],
    [177.5, 171.7], [106.5, 314.3], [121.5, 147], [164.5, 225], [190.5, 118.2],
    [108.5, 260], [196.2, 260], [232.2, 224.7] , [222.2, 160], [162.2, 325],
    [202.2, 345], [227.2, 300], [275.3,172], [ 265.3,172], [ 238.5,300], 
])

Z = np.array([
    74.5, 68.3, 118.25, 113.25, 48.1, 91.75, 40.75, 33, 29, 38, 33, 31, 28.9, 148.75,
    147.75, 127, 74, 76.55, 50.95, 80.55, 118.25, 64, 110.25, 61.55, 99.25, 125.8, 127, 91.8,
    153, 183.5, 166.6, 117.2, 104.2, 164.5
])

# Fit polynomial regression
model, poly = fit_polynomial_regression(X, Z, degree=2)

# Generate predictions
Z_pred = model.predict(poly.transform(X))

# Compute errors and R^2 value
rmse = np.sqrt(mean_squared_error(Z, Z_pred))
mae = mean_absolute_error(Z, Z_pred)
r2 = r2_score(Z, Z_pred)

# Print regression equation and errors
print("Polynomial Regression Equation:")
print(polynomial_equation(model, poly, precision=6))
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.6f}")

# Plot data and regression surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Z, color='red', label='Data Points')

# Create mesh grid for surface plot
x_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 30)
y_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 30)
x_grid, y_grid = np.meshgrid(x_range, y_range)
x_flat, y_flat = x_grid.ravel(), y_grid.ravel()
X_grid = np.column_stack((x_flat, y_flat))
Z_grid = model.predict(poly.transform(X_grid)).reshape(x_grid.shape)

# Plot surface
ax.plot_surface(x_grid, y_grid, Z_grid, alpha=0.5, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Polynomial Regression')
plt.legend()
plt.show()
