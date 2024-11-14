import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('test_data/Wroclaw.csv', sep=';', parse_dates=['date'])

for column in ["temp_pow", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr"]:
    df[column] = df[column].str.replace(",", ".").astype(float)

df = df.dropna(subset=["temp_pow"])
df = df.fillna(df.backfill())

X = df[["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"]]
y = df["temp_pow"]

X = X.fillna(X.backfill())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  
    ('ridge', Ridge())  
])

param_grid = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 100],  
    'poly__degree': [1, 2, 3]  
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

df["temp_pow_pred"] = best_model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['temp_pow'], label='Faktyczna temperatura', color='blue')
plt.plot(df['date'], df['temp_pow_pred'], label='Przewidywania', color='red', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.title('Regresja grzbietowa')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

y_pred_test = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print("Test MSE:", mse)
print("Test R^2 Score:", r2)