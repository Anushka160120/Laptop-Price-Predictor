import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

def convert_memory(mem):
    mem = mem.replace('GB', '').replace('TB', '000').replace('+', ' ')
    parts = mem.split()
    return sum(int(p) for p in parts if p.isdigit())

def preprocessing_data(df):
    df = df.copy()
    df.dropna(inplace=True)
    df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
    df['Memory'] = df['Memory'].apply(convert_memory)
    df.drop(['laptop_ID', 'Product'], axis=1, errors='ignore', inplace=True)
    return df

def load_and_train():

    df = pd.read_csv('laptop_price1.csv', encoding='ISO-8859-1')
    df = preprocessing_data(df)

    x = df.drop('Price_euros', axis=1)
    y = df['Price_euros']

    cat_cols = x.select_dtypes(include='object').columns.tolist()
    num_cols = x.select_dtypes(exclude='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return pipe, cat_cols, num_cols, r2, rmse

def predict_price(pipe, input_dict):
    input_df = pd.DataFrame([input_dict])
    return pipe.predict(input_df)[0]
