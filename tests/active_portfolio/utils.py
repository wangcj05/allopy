def get_id(horizon: int, overweight: bool):
    return f"{horizon}Y{'O' if overweight else 'NO'}"
