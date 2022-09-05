def is_value_type_valid_for_exact_conditions(self, value):
    """ Method to validate if the value is valid for exact match type evaluation.

    Args:
      value: Value to validate.

    Returns:
      Boolean: True if value is a string, boolean, or number. Otherwise False.
    """
    # No need to check for bool since bool is a subclass of int
    if isinstance(value, string_types) or isinstance(value, (numbers.Integral, float)):
      return True

    return False