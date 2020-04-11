class ErrorCalculator:
    obs_var = 1.05
    pred_var = 1.95

    def __init__(self, resid, stand_res, mse, rmse):
        self.resid = resid
        self.stand_res = stand_res
        self.mse = mse
        self.rmse = rmse

    def get_residuals(self):
        self.resid = int(self.obs_var - self.pred_var)
