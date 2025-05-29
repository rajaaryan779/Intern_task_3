### Importing Necessary Libraries ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### Read database ###
df = pd.read_csv("Housing.csv")

### Encode categorical variables ###
df_encoded = pd.get_dummies(df, drop_first=True)

### Define features and target ###
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

### Split data ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Train model ###
model = LinearRegression()
model.fit(X_train, y_train)

### Predictions ###
y_pred = model.predict(X_test)

### Evaluation ###
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): ₹{mae:,.0f}")
print(f"Mean Squared Error (MSE): ₹{mse:,.0f}")
print(f"R² Score: {r2:.2f}")

### Feature Coefficients ###
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Coefficients:\n", coefficients)

### Plot 1: Feature Importance ###
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='coolwarm')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

### Plot 2: Simple Linear Regression (area vs price) ###
X_area = df[['area']]
y_price = df['price']
simple_model = LinearRegression()
simple_model.fit(X_area, y_price)

plt.figure(figsize=(10, 6))
plt.scatter(X_area, y_price, color='blue', label='Actual Price')
plt.plot(X_area, simple_model.predict(X_area), color='red', label='Regression Line')
plt.title('Simple Linear Regression: Area vs Price')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (₹)')
plt.legend()
plt.tight_layout()
plt.show()