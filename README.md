# Polynomial Regression: Age vs Income

## Overview

This project demonstrates the use of polynomial regression to model the relationship between age and income. The script generates synthetic data that simulates a non-linear (quadratic) relationship between these variables and applies polynomial regression to predict income based on age.

## Features

The script includes the following key steps:
- **Synthetic Data Generation**: Creates a dataset with a quadratic relationship between age and income, including random noise to simulate real-world data variability.
- **Data Splitting**: Divides the data into training and testing sets to evaluate model performance.
- **Polynomial Feature Transformation**: Converts the age data into polynomial features of degree 4, enabling the linear regression model to fit a non-linear relationship.
- **Model Training**: Fits a linear regression model using the transformed polynomial features.
- **Prediction and Visualization**: Predicts income for both the training and testing datasets and visualizes the results with a plot.

## Requirements

- Python 3.x
- `numpy` for numerical operations
- `matplotlib` for plotting
- `scikit-learn` for polynomial feature transformation and linear regression

You can install the required packages using pip:
```bash
pip install numpy matplotlib scikit-learn
