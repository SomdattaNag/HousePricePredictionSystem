import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv('data.csv')

# Check for missing values
if df.isnull().sum().any():
    print("Missing values found. Filling missing values with the mean.")
    df = df.fillna(df.mean())  # Filling missing values with the mean

# Define features (X) and target (y)
x = df[['Size (sq ft)', 'Bedrooms', 'Age (years)']]
y = df['Price ($)']

feature_name=x.columns
# Visualizing data distribution
sns.pairplot(df)
plt.show()

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Feature engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_scaled)

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x_poly, y, test_size=0.3, random_state=1)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR(kernel='rbf')
}

best_model = None
best_score = float('-inf')
best_model_name = ""

# Cross-validation and model comparison
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    score = r2_score(ytest, ypred)
    print(f"R² Score for {name}: {score:.4f}")
    
    # Track the best model
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

# Print the best model
print(f"Best model is {best_model_name} with R² Score: {best_score:.4f}")

# Visualize actual vs predicted values for the best model
ypred_best = best_model.predict(xtest)
plt.scatter(ytest, ypred_best)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red')  # Perfect prediction line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'{best_model_name} - Actual vs Predicted')
plt.show(block=False)

# Save the best model for later use
joblib.dump(best_model, 'house_price_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_transform.pkl')  # Save polynomial transformer

# Get user input for prediction
def get_user_input():
    
    n = int(input('Enter number of samples : '))
    sample = []
    for i in range(n):
        a = []
        for feature_name in ['Size (sq ft)', 'Bedrooms', 'Age (years)']:
            ele = float(input(f'Enter data for sample {i+1} {feature_name}: '))
            a.append(ele)
        sample.append(a)
    return np.array(sample)

# Predict new data using the saved model
def make_prediction(input_data):
    # Load model, scaler, and polynomial transformer
    model = joblib.load('house_price_predictor_model.pkl')
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly_transform.pkl')

    # Scale the input data using the same scaler
    input_scaled = scaler.transform(input_data)

    # Apply polynomial transformation
    input_poly = poly.transform(input_scaled)

    # Predict the target values for the scaled input data
    predictions = model.predict(input_poly)

    for price in predictions:
        print(f"Predicted Price: ${price:.2f}")

# Main prediction flow
user_input_data = get_user_input()
make_prediction(user_input_data)
