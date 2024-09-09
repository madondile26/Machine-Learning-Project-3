import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data for demonstration
np.random.seed(0)
ages = np.random.randint(20, 70, 100)  # Random ages between 20 and 70
incomes = 20000 + 500 * ages - 3 * (ages ** 2) + np.random.normal(0, 10000, 100)  # Quadratic relationship with some noise

# Reshape the data for scikit-learn
ages = ages.reshape(-1, 1)
incomes = incomes.reshape(-1, 1)

# Split the data into training and testing sets
split_index = int(0.8 * len(ages))
X_train, X_test = ages[:split_index], ages[split_index:]
y_train, y_test = incomes[:split_index], incomes[split_index:]

# Polynomial regression with degree 3
poly_features = PolynomialFeatures(degree=4)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict incomes for training and testing data
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(np.sort(X_train, axis=0), model.predict(poly_features.transform(np.sort(X_train, axis=0))), color='red', label='Polynomial Regression (Degree=3)')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Polynomial Regression: Age vs Income')
plt.legend()
plt.show()
