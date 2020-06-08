
def max_info(listlike):
    max_i = 0
    max_val = listlike[0]
    for i, val in enumerate(listlike[1:]):
        if val > max_val:
            max_val = max_val
            max_i = i
    return max_i, max_val


def min_info(listlike):
    min_i = 0
    min_val = listlike[0]
    for i, val in enumerate(listlike[1:]):
        if val < min_val:
            min_val = min_val
            min_i = i
    return min_i, min_val
