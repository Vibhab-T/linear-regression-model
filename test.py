import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import linear_regression_class as lr


df = pd.read_csv("ecommerce.csv")

x = df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = df["Yearly Amount Spent"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

linear_model = lr.LinearRegressionModel()

linear_model.fit(x_train, y_train)

predictions = linear_model.predict(x_test)
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel("Predictions")
plt.title("Prediction Vs Actual Value")
plt.show()

errors = linear_model.errors(x_test, y_test)

print("Errors: \n")

print("Mean Absolute Error: ", errors["mean_absolute_error"])
print("Mean Squared Error: ", errors["mean_squared_error"])
print("Root Mean Squared Error: ", errors["root_mse"])
print("R Squared: ", errors["r_squared"])