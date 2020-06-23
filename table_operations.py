from typing import (
    Any,
    List,
    Tuple,
)


def min_aggregation(
    data: List[Tuple[Any]],
    key_fxn,
    value_fxn,
):
    '''
    This is the equivalent of the sql statement:
        select
            a,
            b,

            min(x)
        from data
        group by a,b

    (a, b) as :key_fxn
    x is :value_fxn
    '''
    min_by_key = {}
    for row in data:
        key = key_fxn(row)
        value = value_fxn(row)
        if key in min_by_key:
            if value < min_by_key[key]:
                min_by_key[key] = value
        else:
            min_by_key[key] = value

    out_rows = []
    for key, val in min_by_key.items():
        out_rows.append(key + (val,))
    return out_rows


def group_by(
    data: List[Tuple[Any]],
    key_fxn,
    values_fxn,
):
    by_key = {}
    for row in data:
        key = key_fxn(row)
        value = values_fxn(row)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(value)
    return by_key
