import matplotlib.pyplot as plt
from plotter import Plotter
import pandas as pd


class ScatterPlotter(Plotter):
    def __init__(self, y, y_pred):
        super().__init__(y, y_pred)

    def scatter(self):
        pt = pd.DataFrame({"y_test": self.y, "y_pred": self.y_pred})
        pt.plot.scatter(x='y_test', y='y_pred')
        plt.xlabel('Observed values')
        plt.ylabel('predicted values')
        plt.title('Observed values vs predicted values')
