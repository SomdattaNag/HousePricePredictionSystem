# HousePricePredictionSystem
This is a simple House Price Prediction system Using Machine Learning in Python

# Overview
The House Price Prediction System is a machine learning-based application that predicts the price of a house based on user input features such as size (in square feet), the number of rooms, and the age  (in years). The system uses various regression models to compare performance and provide accurate predictions.

# Features
1. Predict house prices based on user-provided inputs.
2. Support for three different machine learning models:
        a.Linear Regression
        b.Random Forest Regressor
        c.Support Vector Regressor (SVR)
3.Comparison of model performances using metrics such as R² Score.
4.Interactive data visualization of actual vs. predicted prices.

# Usage
1. Input the required features for house prediction:
      a. Size (sq ft)
      b. Number of Rooms
      c. Age (years)
2. Submit to get the predicted price based on the best-performing model.
3. Visualize the actual vs. predicted prices for further analysis.

# Dataset
The dataset should include the following columns:
1. Size (sq ft) : The size of the house in square feet.
2. Rooms : The number of bedrooms in the house.
3. Age (years) : The age of the house in years.
4. Price ($) : The actual price of the house in dollars

# Dependencies:
1.Python 3.8+
2. scikit-learn
3. numpy
4. pandas
5. matplotlib
6. seaborn
7. joblib
