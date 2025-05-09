Made a Linear Regression Class.

Can be imported into other linear regression projects.

While I understand the code and the math is not fully optimal and fast, this was a learning experience.

Instead of just using another library for linear regression implementation, I thought I would create my own model so as to better understand the workings of a linear regression model.

currently, the model only has two methods.

_fit(self, x: pd.DataFrame, y: pd.Series):_

fits a best fit line to the given data, using the least squares method.

For better consistency, both univariate and multivariate calculations could be done with matrix operations but I did not.

_predict(self, x: pd.DataFrame):_

pretty straight forward.
Predicts y based on x.
what the linear model is supposed to do.
