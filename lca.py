import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


housing = pd.read_csv("housing.csv")

print("California Housing Dataset")
print("---------------------------")
print("Original dataset shape:", housing.shape)

housing = housing.dropna()

print("Dataset shape after removing missing values:", housing.shape)

print("\nSample data:")
print(housing.head())


housing = pd.get_dummies(
    housing,
    columns=["ocean_proximity"],
    drop_first=True
)

print("\nDataset after preprocessing:")
print(housing.head())


X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


X_simple = X[["median_income"]]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple,
    y,
    test_size=0.2,
    random_state=42
)

simple_lr = LinearRegression()

simple_lr.fit(X_train_simple, y_train_simple)

simple_pred = simple_lr.predict(X_test_simple)

simple_mae = mean_absolute_error(y_test_simple, simple_pred)
simple_mse = mean_squared_error(y_test_simple, simple_pred)
simple_rmse = np.sqrt(simple_mse)
simple_r2 = r2_score(y_test_simple, simple_pred)


print("\n\nSimple Linear Regression Results")
print("--------------------------------")
print("Slope:", round(simple_lr.coef_[0], 4))
print("Intercept:", round(simple_lr.intercept_, 4))
print("MAE:", round(simple_mae, 4))
print("MSE:", round(simple_mse, 4))
print("RMSE:", round(simple_rmse, 4))
print("R2 Score:", round(simple_r2, 4))


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test_simple["median_income"],
    y_test_simple,
    alpha=0.3,
    label="Actual values"
)

order = np.argsort(X_test_simple["median_income"].values)

plt.plot(
    X_test_simple["median_income"].values[order],
    simple_pred[order],
    linewidth=2,
    label="Regression line"
)

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression")
plt.legend()

plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

multiple_lr = LinearRegression()

multiple_lr.fit(X_train, y_train)

multiple_pred = multiple_lr.predict(X_test)

multiple_mae = mean_absolute_error(y_test, multiple_pred)
multiple_mse = mean_squared_error(y_test, multiple_pred)
multiple_rmse = np.sqrt(multiple_mse)
multiple_r2 = r2_score(y_test, multiple_pred)


print("\n\nMultiple Linear Regression Results")
print("----------------------------------")
print("Intercept:", round(multiple_lr.intercept_, 4))

print("\nFeature Coefficients:")

for name, value in zip(X.columns, multiple_lr.coef_):
    print(name, ":", round(value, 4))

print("\nModel Performance:")
print("MAE:", round(multiple_mae, 4))
print("MSE:", round(multiple_mse, 4))
print("RMSE:", round(multiple_rmse, 4))
print("R2 Score:", round(multiple_r2, 4))


plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    multiple_pred,
    alpha=0.3,
    label="Predicted values"
)


minimum = min(y_test.min(), multiple_pred.min())
maximum = max(y_test.max(), multiple_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Multiple Linear Regression")
plt.legend()

plt.show()


sample = X_test.iloc[[0]]

predicted_value = multiple_lr.predict(sample)[0]
actual_value = y_test.iloc[0]

print("\n\nSample Prediction")
print("-----------------")
print("Actual house value:", round(actual_value, 3))
print("Predicted house value:", round(predicted_value, 3))
print(
    "Prediction error:",
    round(abs(actual_value - predicted_value), 3)
)


results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "MAE": [
        simple_mae,
        multiple_mae
    ],
    "MSE": [
        simple_mse,
        multiple_mse
    ],
    "RMSE": [
        simple_rmse,
        multiple_rmse
    ],
    "R2 Score": [
        simple_r2,
        multiple_r2
    ]
})


print("\n\nFinal Comparison")
print("----------------")
print(results.round(4))


if multiple_r2 > simple_r2:

    print("\nConclusion:")
    print(
        "Multiple Linear Regression performed better than "
        "Simple Linear Regression."
    )

else:

    print("\nConclusion:")
    print(
        "Simple Linear Regression performed better than "
        "Multiple Linear Regression."
    )