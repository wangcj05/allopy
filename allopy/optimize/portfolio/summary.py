import pandas as pd
from statsmodels.iolib.summary2 import Summary

from allopy.optimize.base.result import Result


class PortfolioSummary(Summary):
    def __init__(self, result: Result):
        super().__init__()
        self.result = result

        self.add_title("Portfolio Optimizer")

        if self.result.x is None:
            self.add_text("Problem has not been optimized yet")
            return

        self._add_final_weights()
        self._add_objective_value()
        self._add_constraint_values()

        self.add_text("Optimization completed successfully")

    def _add_final_weights(self):
        assets = [f"Asset {i + 1}" for i in range(len(self.result.x))]
        self.add_df(pd.DataFrame({"Weight": self.result.x, "Assets": assets}))

    def _add_objective_value(self):
        self.add_text(f"Objective Value: {self.result.x:.4}")

    def _add_constraint_values(self):
        self.add_df(pd.DataFrame(self.result.constraint_values))
