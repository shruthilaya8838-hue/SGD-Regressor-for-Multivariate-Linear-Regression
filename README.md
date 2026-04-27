# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Numpy ,Matplotlib and linear regression model
2. Convert x into a 2D array for model training
3. Initialize the linear regression model
4. Fit the model using X and Y
5. Print The predicted marks

## Program:
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]

y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```
## Output:
<img width="1246" height="261" alt="image" src="https://github.com/user-attachments/assets/b38e52b4-429b-411c-852d-52f3dfd19292" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
