def page(self, table_name, paging, constraints=None, *, columns=None, order_by=None,
           get_count=True):
    """Performs a find_all method with paging.

    :param table_name: the name of the table to search on
    :param paging: is a tuple containing (page, page_size).
    :param constraints: is any construct that can be parsed by SqlWriter.parse_constraints.
    :param columns: either a string or a list of column names
    :param order_by: the order by clause
    :param get_count: if True, the total number of records that would be included without paging are
                      returned. If False, None is returned for the count.
    :return: a 2-tuple of (records, total_count)
    """
    if get_count:
      count = self.count(table_name, constraints)
    else:
      count = None

    page, page_size = paging

    limiting = None
    if page_size > 0:
      limiting = (page_size, page * page_size)

    records = list(self.find_all(
      table_name, constraints, columns=columns, order_by=order_by, limiting=limiting))
    return (records, count)