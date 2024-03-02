def average_dict_of_dicts_values_by_key(main_dict: dict, key):
    count = 0
    total = 0
    for dct in main_dict.values():
        val = dct.get(key)
        if val is None:
            continue
        count += 1
        total += val
    if count == 0:
        return 0
    return total / count
