import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
import os
from api_requests import make_gapi_request

# Load the data
data = pd.read_csv("future-gc00-daily-prices.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].values.astype(np.int64) // 10 ** 9
# print(data.Date.head(5))

# Create a directory for saving plots if it doesn't exist
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# Shift the 'Close' column one day back
data['Close'] = data['Close'].shift(-1)

# Drop the last row
data = data[:-1]

# Clean numeric columns by removing commas and converting to numeric type
numeric_columns = ['Open', 'High', 'Low']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col].replace({',': '', '\.': ''}, regex=True), errors='coerce')

data['Close'] = pd.to_numeric(data['Close'].replace({',': '', '\.': ''}, regex=True), errors='coerce')
# print("Unique values in 'Close' column:", data['Close'].unique())

# Features
X = data[['Date', 'Open', 'High', 'Low']]

# Target variable
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Principal Component Analysis
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
predicted = lm.predict(X_test)


# Ridge Regression
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

# Lasso Regression
la = Lasso(alpha=0.5, max_iter=10000) 
la.fit(X_train, y_train)

# Elastic Net Regression
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)


pca = PCA()
pca.fit(X_train)
X_train_hat = pca.transform(X_train)

N = 20
X_train_hat_PCA = X_train_hat[:, :N]  # Assign the transformed data to X_train_hat_PCA

# Now you can subset it
X_train_hat_PCA = X_train_hat_PCA[:, :N]

enet_pca = ElasticNet(tol=0.2, alpha=0.1, l1_ratio=0.1)
enet_pca.fit(X_train_hat_PCA, y_train)

def get_R2_features(model, test=True, save_plot=False, plot_name=None): 
    # Evaluate R^2 for each feature
    features = X_train.columns.tolist()
    R_2_train = []
    R_2_test = []

    for feature in features:
        model_clone = clone(model)  # Create a copy of the model
        model_clone.fit(X_train[[feature]], y_train)
        R_2_test.append(model_clone.score(X_test[[feature]], y_test))
        R_2_train.append(model_clone.score(X_train[[feature]], y_train))

    # Plotting the results
    fig, ax = plt.subplots()  # Create a single figure and axis

    # Set the bar width
    bar_width = 0.35
    index = np.arange(len(features))

    # Plot training R^2
    bar_train = ax.bar(index, R_2_train, bar_width, label="Train")

    # Plot testing R^2 next to training R^2
    bar_test = ax.bar(index + bar_width, R_2_test, bar_width, label="Test")

    plt.xticks(index + bar_width / 2, features, rotation=90)
    plt.ylabel("$R^2$")

    # Combine legends for both training and testing into one
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    # Save the plot instead of showing it
    if save_plot:
        plt.savefig(os.path.join(save_dir, plot_name + ".png"))
    else:
        plt.show()
    
    print("Mean Training R^2 value:", np.mean(R_2_train))
    print("Mean Testing R^2 value:", np.mean(R_2_test))
    print("Max Training R^2 value:", np.max(R_2_train))
    print("Max Testing R^2 value:", np.max(R_2_test))
    print("\n")


def plot_coef(X, model, test=True, save_plot=False, plot_name=None):
    # Plotting coefficients with a single color
    plt.bar(X.columns, abs(model.coef_), color='blue')
    plt.xticks(rotation=90)
    plt.ylabel("$coefficients$")
    plt.title(plot_name)
    plt.legend(["Coefficients"])

    # Save the plot instead of showing it
    if save_plot:
        plt.savefig(os.path.join(save_dir, plot_name + ".png"))
    else:
        plt.show()

    if test:
        print("R^2 on training data:", model.score(X_train, y_train))
        print("R^2 on testing data:", model.score(X_test, y_test))
        print("\n")



print("Linear Regression")
get_R2_features(lm, save_plot=True, plot_name="linear_regression")
plot_coef(X, lm, save_plot=True, plot_name="linear_regression_coef")

print("Ridge Regression")
get_R2_features(rr, save_plot=True, plot_name="ridge_regression")
plot_coef(X, rr, save_plot=True, plot_name="ridge_regression_coef")

print("Lasso Regression")
get_R2_features(la, save_plot=True, plot_name="lasso_regression")
plot_coef(X, la, save_plot=True, plot_name="lasso_regression_coef")

print("Elastic Net Regression")
get_R2_features(enet, save_plot=True, plot_name="elastic_net_regression")
plot_coef(X, enet, save_plot=True, plot_name="elastic_net_regression_coef")


# def get_R2_features(model, X_test, test_labels=None, save_plot=False, plot_name=None):
#     # Check if there are enough samples in the test set
#     if X_test.shape[0] < 2:
#         print("Not enough samples in the test set to calculate R^2.")
#         return

#     # Evaluate R^2 for each feature
#     features = X_test.columns.tolist()
#     R_2_train = []
#     R_2_test = []

#     for feature in features:
#         model_clone = clone(model)
#         model_clone.fit(X_train[[feature]], y_train)
#         R_2_test.append(model_clone.score(X_test[[feature]], test_labels))
#         R_2_train.append(model_clone.score(X_train[[feature]], y_train))

#     # Plotting the results
#     fig, ax = plt.subplots()

#     # Set the bar width
#     bar_width = 0.35
#     index = np.arange(len(features))

#     # Plot training R^2
#     bar_train = ax.bar(index, R_2_train, bar_width, label="Train")

#     # Plot testing R^2 next to training R^2
#     bar_test = ax.bar(index + bar_width, R_2_test, bar_width, label="Test")

#     plt.xticks(index + bar_width / 2, features, rotation=90)
#     plt.ylabel("$R^2$")

#     # Combine legends for both training and testing into one
#     handles, labels = ax.get_legend_handles_labels()
#     plt.legend(handles, labels)

#     # Save the plot instead of showing it
#     if save_plot:
#         plt.savefig(os.path.join(save_dir, plot_name + ".png"))
#     else:
#         plt.show()

#     print("Mean Training R^2 value:", np.mean(R_2_train))
#     print("Mean Testing R^2 value:", np.nanmean(R_2_test))  # Use np.nanmean to handle NaN values
#     print("Max Training R^2 value:", np.max(R_2_train))
#     print("Max Testing R^2 value:", np.nanmax(R_2_test))  # Use np.nanmax to handle NaN values
#     print("\n")

#     # Optionally, print R^2 on testing data if test_labels are provided
#     if test_labels is not None:
#         print("R^2 on testing data:", model.score(X_test, test_labels))
#         print("\n")

        
# def plot_coef(X, model, test=True, save_plot=False, plot_name=None):
#     # Plotting coefficients with a single color
#     plt.bar(X.columns, abs(model.coef_), color='blue')
#     plt.xticks(rotation=90)
#     plt.ylabel("$coefficients$")
#     plt.title(plot_name)
#     plt.legend(["Coefficients"])

#     # Save the plot instead of showing it
#     if save_plot:
#         plt.savefig(os.path.join(save_dir, plot_name + ".png"))
#     else:
#         plt.show()

#     if test:
#         print("R^2 on training data:", model.score(X_train, y_train))
#         print("R^2 on testing data:", model.score(X_test, y_test))
#         print("\n")


# print("Linear Regression")
# get_R2_features(lm, X_test, y_test, save_plot=True, plot_name="linear_regression")
# plot_coef(X, lm, save_plot=True, plot_name="linear_regression_coef")

# print("Ridge Regression")
# get_R2_features(rr,X_test, y_test, save_plot=True, plot_name="ridge_regression")
# plot_coef(X, rr, save_plot=True, plot_name="ridge_regression_coef")

# print("Lasso Regression")
# get_R2_features(la,X_test, y_test, save_plot=True, plot_name="lasso_regression")
# plot_coef(X, la, save_plot=True, plot_name="lasso_regression_coef")

# print("Elastic Net Regression")
# get_R2_features(enet,X_test, y_test, save_plot=True, plot_name="elastic_net_regression")
# plot_coef(X, enet, save_plot=True, plot_name="elastic_net_regression_coef")

# api_df = make_gapi_request()

# # Preprocess the API data (ensure the columns match those used in training)
# X_api = api_df[['Date', 'Open', 'High', 'Low']]
# X_api_scaled = pd.DataFrame(scaler.transform(X_api), columns=X_api.columns)

# # Make predictions for the API data using the trained model
# predicted_api = lm.predict(X_api_scaled)

# # Create a DataFrame with Date, Actual Close Price, and Predicted Close Price
# comparison_df = pd.DataFrame({
#     'Date': api_df['Date'],
#     'Actual_Close': api_df['Close'],
#     'Predicted_Close': predicted_api
# })

# # Print actual and predicted close prices
# print("Comparison of Actual and Predicted Close Prices:")
# print(comparison_df[['Date', 'Actual_Close', 'Predicted_Close']])

# # Plotting actual and predicted close prices side by side
# fig, ax = plt.subplots(figsize=(10, 6))
# bar_width = 0.35
# index = np.arange(len(comparison_df['Date']))

# actual_bars = ax.bar(index, comparison_df['Actual_Close'], bar_width, label='Actual Close Price', color='blue')
# predicted_bars = ax.bar(index + bar_width, comparison_df['Predicted_Close'], bar_width, label='Predicted Close Price', color='orange', alpha=0.7)

# ax.set_xlabel('Date')
# ax.set_ylabel('Close Price')
# ax.set_title('Actual vs Predicted Close Price for API Data')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(comparison_df['Date'])
# ax.legend()

# # Save the plot
# plt.savefig(os.path.join(save_dir, "actual_vs_predicted_close_prices.png"))

# # Show the plot
# plt.show()

# # After making predictions, print some diagnostic information
# print("Comparison of Actual and Predicted Close Prices:")
# print(comparison_df[['Date', 'Actual_Close', 'Predicted_Close']])

# # Additional diagnostic information
# print("\nModel Evaluation on Training Data:")
# print("R^2 on training data:", lm.score(X_train, y_train))

# print("\nModel Evaluation on Testing Data:")
# print("R^2 on testing data:", lm.score(X_test, y_test))