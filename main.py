import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from api_requests import make_gapi_request

try:
    # Load the data
    data = pd.read_csv("future-gc00-daily-prices.csv")
    print("before clean: ", data.head())
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].values.astype(np.int64) // 10 ** 9

    # Save a copy of the original data for later use
    original_data = data.copy()

    # Shift the 'Close' column one day back
    data['Close'] = data['Close'].shift(-1)

    # Drop the last row
    data = data[:-1]

    # Clean numeric columns
    numeric_columns = ['Open', 'High', 'Low']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col].replace({',': '', '\.': ''}, regex=True), errors='coerce')

    data['Close'] = pd.to_numeric(data['Close'].replace({',': '', '\.': ''}, regex=True), errors='coerce')
    print("after clean: ", data.head())

    # Features (X) and Target variable (y)
    X = data[['Date', 'Open', 'High', 'Low', 'Close']]  # Include 'Close' in features
    y = data['Close']

    # Drop 'Date' column before scaling
    X = X.drop(columns=['Date'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=2)  # Adjust the number of components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Linear Regression Model
    model_lr = LinearRegression()
    model_lr.fit(X_train_pca, y_train)

    # Ridge Regression Model
    model_ridge = Ridge()
    model_ridge.fit(X_train_pca, y_train)

    # Lasso Regression Model
    model_lasso = Lasso()
    model_lasso.fit(X_train_pca, y_train)

    # ElasticNet Regression Model
    model_en = ElasticNet()
    model_en.fit(X_train_pca, y_train)

    # Evaluate models on the testing set
    for model, name in zip([model_lr, model_ridge, model_lasso, model_en],
                           ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression']):
        y_pred = model.predict(X_test_pca)
        r2 = r2_score(y_test, y_pred)
        print(f'R2 score for {name}: {r2}')

    # Use your API function to get data for future dates
    future_data = make_gapi_request()

    if future_data is not None:
        # Revert changes to the original data format
        future_data['Date'] = pd.to_datetime(future_data['Date'], unit='s')
        for col in numeric_columns:
            future_data[col] = pd.to_numeric(future_data[col].replace({',': '', '\.': ''}, regex=True), errors='coerce')

        # Include 'Close' in features
        future_data['Close'] = pd.to_numeric(future_data['Close'].replace({',': '', '\.': ''}, regex=True), errors='coerce')

        # Drop 'Date' column before scaling
        future_data = future_data.drop(columns=['Date'])

        # Apply the same cleaning process to the original_data
        for col in numeric_columns:
            original_data[col] = pd.to_numeric(original_data[col].replace({',': '', '\.': ''}, regex=True),
                                               errors='coerce')

        original_data['Close'] = pd.to_numeric(original_data['Close'].replace({',': '', '\.': ''}, regex=True),
                                              errors='coerce')

        # Include 'Close' in features for scaling
        original_data_scaled = scaler.transform(original_data[['Open', 'High', 'Low', 'Close']])
        future_data_scaled = scaler.transform(future_data)

        # Apply PCA to future data
        future_data_pca = pca.transform(future_data_scaled)

        # Predict future prices
        future_price_lr = model_lr.predict(future_data_pca)
        future_price_ridge = model_ridge.predict(future_data_pca)
        future_price_lasso = model_lasso.predict(future_data_pca)
        future_price_en = model_en.predict(future_data_pca)

        # Extract actual close prices from the API response
        api_actual_close = future_data['Close'].values

        # Print predicted and actual close prices for each model
        print(f'Linear Regression - Predicted Close: {future_price_lr}, Actual Close: {api_actual_close}')
        print(f'Ridge Regression - Predicted Close: {future_price_ridge}, Actual Close: {api_actual_close}')
        print(f'Lasso Regression - Predicted Close: {future_price_lasso}, Actual Close: {api_actual_close}')
        print(f'ElasticNet Regression - Predicted Close: {future_price_en}, Actual Close: {api_actual_close}')

    else:
        print("Error in API request. Unable to predict future prices.")

except Exception as e:
    print(f"An error occurred: {e}")
