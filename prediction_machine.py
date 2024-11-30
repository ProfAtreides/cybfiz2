import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Criteria - list of columns in dataset used to predict temperature
# Time_span - obvious
# Debug mode - true give us prints and shows where training data ends on the plot
def make_prediction(criteria, time_span, debug_mode):
    df = pd.read_csv('test_data/Wroclaw.csv', sep=';', parse_dates=['date'])

    for column in ["temp_pow", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr"]:
        df[column] = df[column].str.replace(",", ".").astype(float)

    df = df.dropna(subset=["temp_pow"])
    df = df.fillna(df.backfill())

    X = df[criteria]
    y = df["temp_pow"]

    X = X.fillna(X.backfill())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('poly', PolynomialFeatures(degree=7, include_bias=False)),  
        ('ridge', Ridge())  
    ])

    param_grid = {
        'ridge__alpha': [0.2,0.3,0.35, 0.5,1, 10, 100, 1000],  
        'poly__degree': [1, 2, 2, 3,5]  
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    if debug_mode:
        print("Best parameters:", grid_search.best_params_)
        
    df["temp_pow_pred"] = best_model.predict(X)
    
    train_end_date = df['date'][len(X_train)-1]
    prediction_end_date = train_end_date + pd.Timedelta(days=time_span)

    df_filtered = df[(df['date'] > train_end_date) & (df['date'] <= prediction_end_date)]

    


    plt.figure(figsize=(12, 6))
    if debug_mode:
        plt.plot(df['date'], df['temp_pow'], label='Faktyczna temperatura', color='blue',linestyle='--')
        plt.plot(df['date'], df['temp_pow_pred'], label='Faktyczna temperatura', color='blue',linestyle='--')
    else:    
        plt.plot(df_filtered['date'], df_filtered['temp_pow_pred'], color='red', )
    if debug_mode:
        plt.axvline(x=train_end_date, color='green', linestyle='--', label='Koniec danych treningowych')
    plt.xlabel('Data')
    plt.ylabel('Temperatura')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if debug_mode:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['temp_pow']-df['temp_pow_pred'], label='Temp diff', color='purple')
        plt.axvline(x=train_end_date, color='green', linestyle='--', label='Koniec danych treningowych')
        plt.xlabel('Data')
        plt.ylabel('Temperatura')
        plt.title('Temp diff')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if debug_mode:
        y_pred_test = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        print("Test MSE:", mse)
        print("Test R^2 Score:", r2)
    
columns = ["kier_wiatr", "temp_grunt", "suma_opad_doba", "sr_pred_wiatr", "wilg"]
time_span = 30
debug_mode = False

make_prediction(columns,time_span, debug_mode)