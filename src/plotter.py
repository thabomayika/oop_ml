import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self, y, y_pred, resid=None):
        self.resid = resid
        self.y = y
        self.y_pred = y_pred

    def run_calculations(self):
        resids = np.array(self.y) - np.array(self.y_pred)
        return resids

    def plot(self):
        if self.resid is None:
            self.run_calculations()
        sns.set()
        plt.hist(self.resid)
        plt.title('Distribution of residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Distribution')
