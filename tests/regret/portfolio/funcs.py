from allopy import OptData, get_option

__all__ = ["obj_max_returns", "cvar_fun"]


def obj_max_returns(cube: OptData):
    def obj_fun(w):
        return get_option("F.SCALE") * cube.expected_return(w, True)

    return obj_fun


def cvar_fun(cube: OptData, target: float):
    def cvar(w):
        return get_option("F.SCALE") * (target - cube.cvar(w, True, 5.0))

    return cvar
