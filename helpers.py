def parse_float_value(kv):
    """
    Parse key-value pair divided by = where value is float

    :param kv: key=value string
    :return: (key, float(value)) tuple
    """
    key, value = kv.split('=', 1)
    return key, float(value)
