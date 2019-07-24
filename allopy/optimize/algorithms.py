from typing import Union

from nlopt import *

__all__ = [
    "GN_DIRECT",
    "GN_DIRECT_L",
    "GN_DIRECT_L_RAND",
    "GN_DIRECT_NOSCAL",
    "GN_DIRECT_L_NOSCAL",
    "GN_DIRECT_L_RAND_NOSCAL",
    "GN_ORIG_DIRECT",
    "GN_ORIG_DIRECT_L",
    "GD_STOGO",
    "GD_STOGO_RAND",
    "LD_LBFGS_NOCEDAL",
    "LD_LBFGS",
    "LN_PRAXIS",
    "LD_VAR1",
    "LD_VAR2",
    "LD_TNEWTON",
    "LD_TNEWTON_RESTART",
    "LD_TNEWTON_PRECOND",
    "LD_TNEWTON_PRECOND_RESTART",
    "GN_CRS2_LM",
    "GN_MLSL",
    "GD_MLSL",
    "GN_MLSL_LDS",
    "GD_MLSL_LDS",
    "LD_MMA",
    "LN_COBYLA",
    "LN_NEWUOA",
    "LN_NEWUOA_BOUND",
    "LN_NELDERMEAD",
    "LN_SBPLX",
    "LN_AUGLAG",
    "LD_AUGLAG",
    "LN_AUGLAG_EQ",
    "LD_AUGLAG_EQ",
    "LN_BOBYQA",
    "GN_ISRES",
    "AUGLAG",
    "AUGLAG_EQ",
    "G_MLSL",
    "G_MLSL_LDS",
    "LD_SLSQP",
    "LD_CCSAQ",
    "GN_ESCH",
    "GN_AGS",
    "map_algorithm",
    "has_gradient"
]


def map_algorithm(algorithm):
    """
    Maps the string name of an algorithm to it's nlopt equivalent

    Parameters
    ----------
    algorithm: str
        name of algorithm

    Returns
    -------
    str
        nlopt name
    """
    return {
        'GN_DIRECT': GN_DIRECT,
        'GN_DIRECT_L': GN_DIRECT_L,
        'GN_DIRECT_L_RAND': GN_DIRECT_L_RAND,
        'GN_DIRECT_NOSCAL': GN_DIRECT_NOSCAL,
        'GN_DIRECT_L_NOSCAL': GN_DIRECT_L_NOSCAL,
        'GN_DIRECT_L_RAND_NOSCAL': GN_DIRECT_L_RAND_NOSCAL,
        'GN_ORIG_DIRECT': GN_ORIG_DIRECT,
        'GN_ORIG_DIRECT_L': GN_ORIG_DIRECT_L,
        'GD_STOGO': GD_STOGO,
        'GD_STOGO_RAND': GD_STOGO_RAND,
        'LD_LBFGS_NOCEDAL': LD_LBFGS_NOCEDAL,
        'LD_LBFGS': LD_LBFGS,
        'LN_PRAXIS': LN_PRAXIS,
        'LD_VAR1': LD_VAR1,
        'LD_VAR2': LD_VAR2,
        'LD_TNEWTON': LD_TNEWTON,
        'LD_TNEWTON_RESTART': LD_TNEWTON_RESTART,
        'LD_TNEWTON_PRECOND': LD_TNEWTON_PRECOND,
        'LD_TNEWTON_PRECOND_RESTART': LD_TNEWTON_PRECOND_RESTART,
        'GN_CRS2_LM': GN_CRS2_LM,
        'GN_MLSL': GN_MLSL,
        'GD_MLSL': GD_MLSL,
        'GN_MLSL_LDS': GN_MLSL_LDS,
        'GD_MLSL_LDS': GD_MLSL_LDS,
        'LD_MMA': LD_MMA,
        'LN_COBYLA': LN_COBYLA,
        'LN_NEWUOA': LN_NEWUOA,
        'LN_NEWUOA_BOUND': LN_NEWUOA_BOUND,
        'LN_NELDERMEAD': LN_NELDERMEAD,
        'LN_SBPLX': LN_SBPLX,
        'LN_AUGLAG': LN_AUGLAG,
        'LD_AUGLAG': LD_AUGLAG,
        'LN_AUGLAG_EQ': LN_AUGLAG_EQ,
        'LD_AUGLAG_EQ': LD_AUGLAG_EQ,
        'LN_BOBYQA': LN_BOBYQA,
        'GN_ISRES': GN_ISRES,
        'AUGLAG': AUGLAG,
        'AUGLAG_EQ': AUGLAG_EQ,
        'G_MLSL': G_MLSL,
        'G_MLSL_LDS': G_MLSL_LDS,
        'LD_SLSQP': LD_SLSQP,
        'LD_CCSAQ': LD_CCSAQ,
        'GN_ESCH': GN_ESCH,
        'GN_AGS': GN_AGS
    }[algorithm]


def has_gradient(algorithm) -> Union[str, bool]:
    desc: str = algorithm_name(algorithm)

    if desc.endswith('(NOT COMPILED)'):
        return 'NOT COMPILED'

    if 'no-derivative' in desc:
        return False

    return 'derivative-based' in desc or 'derivative)' in desc
