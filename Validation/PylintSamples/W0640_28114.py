def _PreparedData(self, order_by=()):
    """Prepares the data for enumeration - sorting it by order_by.

    Args:
      order_by: Optional. Specifies the name of the column(s) to sort by, and
                (optionally) which direction to sort in. Default sort direction
                is asc. Following formats are accepted:
                "string_col_name"  -- For a single key in default (asc) order.
                ("string_col_name", "asc|desc") -- For a single key.
                [("col_1","asc|desc"), ("col_2","asc|desc")] -- For more than
                    one column, an array of tuples of (col_name, "asc|desc").

    Returns:
      The data sorted by the keys given.

    Raises:
      DataTableException: Sort direction not in 'asc' or 'desc'
    """
    if not order_by:
      return self.__data

    sorted_data = self.__data[:]
    if isinstance(order_by, six.string_types) or (
        isinstance(order_by, tuple) and len(order_by) == 2 and
        order_by[1].lower() in ["asc", "desc"]):
      order_by = (order_by,)
    for key in reversed(order_by):
      if isinstance(key, six.string_types):
        sorted_data.sort(key=lambda x: x[0].get(key))
      elif (isinstance(key, (list, tuple)) and len(key) == 2 and
            key[1].lower() in ("asc", "desc")):
        key_func = lambda x: x[0].get(key[0])
        sorted_data.sort(key=key_func, reverse=key[1].lower() != "asc")
      else:
        raise DataTableException("Expected tuple with second value: "
                                 "'asc' or 'desc'")

    return sorted_data