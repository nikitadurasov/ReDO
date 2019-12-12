def parse_float_value(kv):
    key, value = kv.split('=', 1)
    return key, float(value)
