import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Test1", "Test2", "assets", "scenarios", "outlook"]

assets = 'DMEQ', 'EMEQ', 'PE', 'RE', 'NB', 'EILB', 'CASH'
scenarios = 'Baseline', 'Goldilocks', 'Stagflation', 'HHT'
outlook = pd.read_csv(os.path.join(os.path.dirname(__file__), "scenario.csv"))


@dataclass
class Bounds:
    DMEQ: float
    EMEQ: float
    PE: float
    RE: float
    NB: float
    EILB: float
    CASH: float

    def as_array(self):
        return [self.DMEQ, self.EMEQ, self.PE, self.RE, self.NB, self.EILB, self.CASH]


@dataclass
class CVAR:
    Baseline: float
    Goldilocks: float
    Stagflation: float
    HHT: float

    def as_array(self):
        return np.array([self.Baseline, self.Goldilocks, self.Stagflation, self.HHT])

    def __getitem__(self, item: str):
        return self.__dict__[item]


@dataclass
class Probability(CVAR):
    pass


class Weights(Bounds):
    pass


class OptimalMix(CVAR):
    pass


@dataclass
class Expected:
    Mix: OptimalMix
    Baseline: Weights
    Goldilocks: Weights
    Stagflation: Weights
    HHT: Weights
    Optimal: Weights


@dataclass
class RegretTest:
    lb: Bounds
    ub: Bounds
    cvar: CVAR
    prob: Probability
    expected: Expected

    @property
    def solutions(self):
        return np.array([
            self.expected.Baseline.as_array(),
            self.expected.Goldilocks.as_array(),
            self.expected.Stagflation.as_array(),
            self.expected.HHT.as_array(),
        ])

    @property
    def optimal(self):
        return self.expected.Optimal.as_array()

    @property
    def proportions(self):
        return self.expected.Mix.as_array()


Test1 = RegretTest(
    lb=Bounds(
        DMEQ=0,
        EMEQ=0,
        PE=0.13,
        RE=0.11,
        NB=0,
        EILB=0.05,
        CASH=0.04,
    ),
    ub=Bounds(
        DMEQ=1,
        EMEQ=0.18,
        PE=0.13,
        RE=0.11,
        NB=1,
        EILB=0.05,
        CASH=0.04,
    ),
    prob=Probability(
        Baseline=0.57,
        Goldilocks=0.1,
        Stagflation=0.14,
        HHT=0.19,
    ),
    cvar=CVAR(
        Baseline=-0.34,
        Goldilocks=-0.253,
        Stagflation=-0.501,
        HHT=-0.562
    ),
    expected=Expected(
        Mix=OptimalMix(
            Baseline=0.732,
            Goldilocks=0,
            Stagflation=0,
            HHT=0.268
        ),
        Baseline=Weights(
            DMEQ=0.25,
            EMEQ=0.18,
            PE=0.13,
            RE=0.11,
            NB=0.24,
            EILB=0.05,
            CASH=0.04,
        ),
        Goldilocks=Weights(
            DMEQ=0.3485,
            EMEQ=0.0947,
            PE=0.13,
            RE=0.11,
            NB=0.2268,
            EILB=0.05,
            CASH=0.04,
        ),
        Stagflation=Weights(
            DMEQ=0.1,
            EMEQ=0,
            PE=0.13,
            RE=0.11,
            NB=0.57,
            EILB=0.05,
            CASH=0.04,
        ),
        HHT=Weights(
            DMEQ=0,
            EMEQ=0,
            PE=0.13,
            RE=0.11,
            NB=0.67,
            EILB=0.05,
            CASH=0.04,
        ),
        Optimal=Weights(
            DMEQ=0.183,
            EMEQ=0.132,
            PE=0.13,
            RE=0.11,
            NB=0.355,
            EILB=0.05,
            CASH=0.04,
        ),
    )
)

Test2 = RegretTest(
    lb=Bounds(
        DMEQ=0,
        EMEQ=0,
        PE=0,
        RE=0,
        NB=0,
        EILB=0,
        CASH=0,
    ),
    ub=Bounds(
        DMEQ=1,
        EMEQ=0.18,
        PE=0.13,
        RE=0.11,
        NB=1,
        EILB=0.05,
        CASH=0.04,
    ),
    prob=Probability(
        Baseline=0.57,
        Goldilocks=0.1,
        Stagflation=0.14,
        HHT=0.19,
    ),
    cvar=CVAR(
        Baseline=-0.34,
        Goldilocks=-0.253,
        Stagflation=-0.501,
        HHT=-0.562
    ),
    expected=Expected(
        Mix=OptimalMix(
            Baseline=0.765,
            Goldilocks=0,
            Stagflation=0,
            HHT=0.235
        ),
        Baseline=Weights(
            DMEQ=0.25,
            EMEQ=0.18,
            PE=0.13,
            RE=0.11,
            NB=0.24,
            EILB=0.05,
            CASH=0.04,
        ),
        Goldilocks=Weights(
            DMEQ=0.357,
            EMEQ=0.099,
            PE=0.13,
            RE=0.11,
            NB=0.264,
            EILB=0,
            CASH=0.04,
        ),
        Stagflation=Weights(
            DMEQ=0.1,
            EMEQ=0,
            PE=0.13,
            RE=0.11,
            NB=0.57,
            EILB=0.05,
            CASH=0.04,
        ),
        HHT=Weights(
            DMEQ=0,
            EMEQ=0,
            PE=0,
            RE=0,
            NB=0.95,
            EILB=0.05,
            CASH=0,
        ),
        Optimal=Weights(
            DMEQ=0.191,
            EMEQ=0.138,
            PE=0.1,
            RE=0.084,
            NB=0.407,
            EILB=0.05,
            CASH=0.031,
        ),
    )
)
