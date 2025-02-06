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
df = pd.read_csv('data.csv')
if df.isnull().sum().any():
    print("Missing values found. Filling missing values with the mean.")
    df = df.fillna(df.mean())  
x = df[['Size (sq ft)', 'Rooms', 'Age (years)']]
y = df['Price ($)']
sns.pairplot(df)
plt.show()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_scaled)
xtrain, xtest, ytrain, ytest = train_test_split(x_poly, y, test_size=0.3, random_state=1)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR(kernel='rbf')
}
best_model = None
best_score = float('-inf')
best_model_name = ""
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    score = r2_score(ytest, ypred)
    print(f"R² Score for {name}: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"Best model is {best_model_name} with R² Score: {best_score:.4f}")
ypred_best = best_model.predict(xtest)
plt.scatter(ytest, ypred_best)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], color='red')  
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'{best_model_name} - Actual vs Predicted')
plt.show(block=False)
joblib.dump(best_model, 'house_price_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_transform.pkl')  
def get_user_input():
    n = int(input('Enter number of samples : '))
    sample = []
    for i in range(n):
        a = []
        for feature_name in ['Size (sq ft)', 'Rooms', 'Age (years)']:
            ele = float(input(f'Enter data for sample {i+1} {feature_name}: '))
            a.append(ele)
        sample.append(a)
    return np.array(sample)
def make_prediction(input_data):
    model = joblib.load('house_price_predictor_model.pkl')
    scaler = joblib.load('scaler.pkl')
    poly = joblib.load('poly_transform.pkl')
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)
    predictions = model.predict(input_poly)
    for price in predictions:
        print(f"Predicted Price: ${price:.2f}")
userinput = get_user_input()
make_prediction(userinput)
