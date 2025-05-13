import pandas as pd
import numpy as np
from scipy import stats

class LinearRegressionModel:
    
    def __init__(self):
        self.coeffs = None
        self.intercepts = None
        self.single_variate = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        
        if x.shape[1] == 1:
            X = x.iloc[:, 0].to_numpy()
            Y = y.to_numpy()
            N = len(X)
            sum_x = np.sum(X)
            sum_y = np.sum(Y)
            xy = X * Y
            x2 = X * X
            sum_x2 = np.sum(x2)
            sum_xy = np.sum(xy)

            #least squares 
            gradient = ((N*sum_xy)-(sum_x*sum_y))/((N*sum_x2)-(sum_x*sum_x))
            intercept = (sum_y - (gradient * sum_x))/N

            self.coeffs = [gradient]
            self.intercepts = [intercept]
            self.single_variate = True
        
        else: 
            self.single_variate = False

            X = x.to_numpy()
            Y = y.to_numpy().reshape(-1, 1) #reshape to a vector
            N = X.shape[0]

            x_bias =  np.hstack((np.ones((N, 1)), X))


           # Ordinary Least Squares Method: θ = (XᵀX)^(-1) Xᵀ y
            theta = np.linalg.inv(x_bias.T @ x_bias) @ x_bias.T @ Y

            self.intercepts = theta[0, 0]
            self.coeffs = theta[1:].flatten()



    def predict(self, x: pd.DataFrame):
        if self.single_variate == True:
            gradient = self.coeffs[0]
            intercept = self.intercepts[0]

            X = x.iloc[:, 0].to_numpy()
            Y = gradient * X + intercept

            return Y
        else:
            X = x.to_numpy()
            N = X.shape[0]

            x_bias = np.hstack((np.ones((N, 1)), X))

            theta = np.concatenate(([self.intercepts], self.coeffs))

            Y = x_bias @ theta

            return Y 
        
    def errors(self, x: pd.DataFrame, y: pd.Series):
        y_prediction = np.array(self.predict(x))
        y = y.to_numpy()

        n = len(y)
        k = x.shape[1]

        mean_absolute_error = np.mean(np.abs(y-y_prediction)) 
        mean_squared_error = np.mean((y - y_prediction) ** 2)
        root_mse = np.sqrt(mean_squared_error)

        ss_res = np.sum((y-y_prediction)**2)
        ss_tot = np.sum((y-np.mean(y)) ** 2)

        r_squared = 1 - (ss_res/ ss_tot)

        if k<n-1 and r_squared < 1:
            f_stat = (r_squared/k) / ((1-r_squared)/(n-k-1))
            p_value = 1 - stats.f.cdf(f_stat, k, n-k-1)
        else:
            f_stat = np.nan
            p_value = np.nan

        errors = {
            "mean_absolute_error" : mean_absolute_error,
            "mean_squared_error": mean_squared_error,
            "root_mse": root_mse,
            "r_squared": r_squared, 
            "f_stat": f_stat,
            "p_value": p_value
        }

        return errors
    
    '''
        Mean Absolute Error = (1/N)(E[1 to n]|yi - y^i|)
        Mean Squared Error = (1/N)(E[1 to n](Yi - y^i)2)
        RMSE = sqrt(Mean Squared Error)


        Residual Sum of Squares, ss_res = =∑(yi - y^i)2
        total sum of squares = ∑(yi - yˉ)2

        R_squared = 1 - (SSres  / SStot)

        F = (R2/k)/((1-R2)/(n-k-1))

    '''