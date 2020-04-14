from sklearn.metrics import mean_squared_error as mse


class ErrorCalculator:

    obs_var = 1.05
    pred_var = 1.95

    def __init__(self, resid, stand_res, mse, rmse):
        self.resid = resid
        self.stand_res = stand_res
        self.mse = mse
        self.rmse = rmse

    def get_residuals(self):
        self.resid = self.obs_var - self.pred_var

    def get_standardised_residuals(self):
        self.stand_res = (self.obs_var - self.pred_var) / (self.pred_var)**0.5

    def get_mse(self):
        self.mse = mse(self.obs_var, self.pred_var)
