from plotter import Plotter


class HistogramPlotter(Plotter):
    def __init__(self, y, y_pred):
        super().__init__(y, y_pred)
