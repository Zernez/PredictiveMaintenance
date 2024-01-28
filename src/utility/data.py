def get_window_size(n_condition):
    if n_condition == 0:
        return 2
    elif n_condition == 1:
        return 4
    return 6

def get_lag(n_condition):
    if n_condition == 0:
        return -1
    elif n_condition == 1:
        return -3
    return -5