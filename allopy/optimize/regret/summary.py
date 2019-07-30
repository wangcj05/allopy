import pandas as pd
from statsmodels.iolib.summary2 import Summary

from .result import RegretResult


class RegretSummary(Summary):
    def __init__(self, result: RegretResult):
        super().__init__()
        self.result = result

        self.add_title("Regret Optimizer")

        if self.result.sol is None:
            self.add_text("Problem has not been optimized yet")
            return

        self._add_first_level_solutions()
        self._add_scenario_proportions()
        self._add_final_weights()

        self.add_text("Optimization completed successfully")

    def _add_first_level_solutions(self):
        self.add_df(pd.DataFrame({
            "Assets": self.result.asset_names,
            **{s: w for s, w in zip(self.result.scenario_names, self.result.first_level_solutions)},
        }))

    def _add_final_weights(self):
        self.add_df(pd.DataFrame({
            "Assets": self.result.asset_names,
            "Weight": self.result.sol,
        }))

    def _add_scenario_proportions(self):
        if self.result.props:
            self.add_df(pd.DataFrame({
                "Scenario": self.result.scenario_names,
                "Proportion (%)": self.result.props.round(4) * 100
            }))
