import pandas as pd
from statsmodels.iolib.summary2 import Summary

from .result import RegretResult


class RegretSummary(Summary):
    def __init__(self, result: RegretResult):
        super().__init__()
        self.result = result

        self.add_title("Regret Optimizer")

        if self.result.solution is None:
            self.add_text("Problem has not been optimized yet")
            return

        self._add_scenario_optimal_weights()
        self._add_scenario_proportions()
        self._add_final_weights()

        self.add_text("Optimization completed successfully")

    def _add_scenario_optimal_weights(self):
        self.add_df(pd.DataFrame(
            {s: w for s, w in zip(self.result.scenarios, self.result.scenario_solutions)},
            self.result.assets)
        )

    def _add_final_weights(self):
        self.add_df(pd.DataFrame({"Weight": self.result.solution}, self.result.assets))

    def _add_scenario_proportions(self):
        if self.result.proportions is not None:
            self.add_df(pd.DataFrame({
                "Scenario": self.result.scenarios,
                "Proportion (%)": self.result.proportions.round(4) * 100
            }))
