from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


class ErrorCalculator:

    y = 1.05
    y_pred = 1.95

    def __init__(self, resid, stand_res, mse, rmse):
        self.resid = resid
        self.stand_res = stand_res
        self.mse = mse
        self.rmse = rmse

    def get_residuals(self):
        self.resid = self.obs_var - self.pred_var
        return self.resid

    def get_standardised_residuals(self):
        self.stand_res = (self.obs_var - self.pred_var) / (self.pred_var)**0.5
        return self.stand_res

    def get_mse(self):
        self.mse = mse(self.obs_var, self.pred_var)
        return self.mse

    def get_rmse(self):
        self.rmse = mse(self.obs_var, self.pred_var, squared=False)
        return self.rmse

    def error_summary(self):
        stand_resid_min = min(self.stand_res)
        stand_resid_max = max(self.stand_res)
        rmse_min = min(self.rmse)
        rmse_max = max(self.rmse)
        mse_min = min(self.mse)
        mse_max = max(self.mse)
        print(f'standard residual: {stand_resid_min}')
        print(f'standard residual: {stand_resid_max}')
        print(f'min rmse: {rmse_min}')
        print(f'max rmse: {rmse_max}')
        print(f'min mse: {mse_min}')
        print(f'max mse: {mse_max}')


class Plotter(ErrorCalculator):

    def plot(self):
        plt.hist()
