import pandas as pd
import numpy as np

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
        
    def rSquared(self, x: pd.DataFrame, y: pd.Series):
        y_prediction = self.predict(x)
        ss_res = np.sum((y-y_prediction)**2)
        ss_tot = np.sum((y-np.mean(y)) ** 2)

        r_squared = 1 - (ss_res/ ss_tot)

        return r_squared
    
    '''
        Residual Sum of Squares, ss_res = =∑(yi - y^i)2
        total sum of squares = ∑(yi - yˉ)2

        R_squared = 1 - (SSres  / SStot)

    '''