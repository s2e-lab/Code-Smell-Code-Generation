def _assert_sframe_equal(sf1,
                         sf2,
                         check_column_names=True,
                         check_column_order=True,
                         check_row_order=True,
                         float_column_delta=None):
    """
    Assert the two SFrames are equal.

    The default behavior of this function uses the strictest possible
    definition of equality, where all columns must be in the same order, with
    the same names and have the same data in the same order.  Each of these
    stipulations can be relaxed individually and in concert with another, with
    the exception of `check_column_order` and `check_column_names`, we must use
    one of these to determine which columns to compare with one another.

    Parameters
    ----------
    sf1 : SFrame

    sf2 : SFrame

    check_column_names : bool
        If true, assert if the data values in two columns are the same, but
        they have different names.  If False, column order is used to determine
        which columns to compare.

    check_column_order : bool
        If true, assert if the data values in two columns are the same, but are
        not in the same column position (one is the i-th column and the other
        is the j-th column, i != j).  If False, column names are used to
        determine which columns to compare.

    check_row_order : bool
        If true, assert if all rows in the first SFrame exist in the second
        SFrame, but they are not in the same order.

    float_column_delta : float
        The acceptable delta that two float values can be and still be
        considered "equal". When this is None, only exact equality is accepted.
        This is the default behavior since columns of all Nones are often of
        float type. Applies to all float columns.
    """
    from .. import SFrame as _SFrame
    if (type(sf1) is not _SFrame) or (type(sf2) is not _SFrame):
        raise TypeError("Cannot function on types other than SFrames.")

    if not check_column_order and not check_column_names:
        raise ValueError("Cannot ignore both column order and column names.")

    sf1.__materialize__()
    sf2.__materialize__()

    if sf1.num_columns() != sf2.num_columns():
        raise AssertionError("Number of columns mismatched: " +
            str(sf1.num_columns()) + " != " + str(sf2.num_columns()))

    s1_names = sf1.column_names()
    s2_names = sf2.column_names()

    sorted_s1_names = sorted(s1_names)
    sorted_s2_names = sorted(s2_names)

    if check_column_names:
        if (check_column_order and (s1_names != s2_names)) or (sorted_s1_names != sorted_s2_names):
            raise AssertionError("SFrame does not have same column names: " +
                str(sf1.column_names()) + " != " + str(sf2.column_names()))

    if sf1.num_rows() != sf2.num_rows():
        raise AssertionError("Number of rows mismatched: " +
            str(sf1.num_rows()) + " != " + str(sf2.num_rows()))

    if not check_row_order and (sf1.num_rows() > 1):
        sf1 = sf1.sort(s1_names)
        sf2 = sf2.sort(s2_names)

    names_to_check = None
    if check_column_names:
      names_to_check = list(zip(sorted_s1_names, sorted_s2_names))
    else:
      names_to_check = list(zip(s1_names, s2_names))
    for i in names_to_check:
        col1 = sf1[i[0]]
        col2 = sf2[i[1]]
        if col1.dtype != col2.dtype:
            raise AssertionError("Columns " + str(i) + " types mismatched.")

        compare_ary = None
        if col1.dtype == float and float_column_delta is not None:
            dt = float_column_delta
            compare_ary = ((col1 > col2-dt) & (col1 < col2+dt))
        else:
            compare_ary = (sf1[i[0]] == sf2[i[1]])
        if not compare_ary.all():
            count = 0
            for j in compare_ary:
                if not j:
                  first_row = count
                  break
                count += 1
            raise AssertionError("Columns " + str(i) +
                " are not equal! First differing element is at row " +
                str(first_row) + ": " + str((col1[first_row],col2[first_row])))