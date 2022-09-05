def least_upper_bound(*intervals_to_join):
        """
        Pseudo least upper bound.
        Join the given set of intervals into a big interval. The resulting strided interval is the one which in
        all the possible joins of the presented SI, presented the least number of values.

        The number of joins to compute is linear with the number of intervals to join.

        Draft of proof:
        Considering  three generic SI (a,b, and c) ordered from their lower bounds, such that
        a.lower_bund <= b.lower_bound <= c.lower_bound, where <= is the lexicographic less or equal.
        The only joins which have sense to compute are:
        * a U b U c
        * b U c U a
        * c U a U b

        All the other combinations fall in either one of these cases. For example: b U a U c does not make make sense
        to be calculated. In fact, if one draws this union, the result is exactly either (b U c U a) or (a U b U c) or
        (c U a U b).
        :param intervals_to_join: Intervals to join
        :return: Interval that contains all intervals
        """
        assert len(intervals_to_join) > 0, "No intervals to join"
        # Check if all intervals are of same width
        all_same = all(x.bits == intervals_to_join[0].bits for x in intervals_to_join)
        assert all_same, "All intervals to join should be same"

        # Optimization: If we have only one interval, then return that interval as result
        if len(intervals_to_join) == 1:
            return intervals_to_join[0].copy()
        # Optimization: If we have only two intervals, the pseudo-join is fine and more precise
        if len(intervals_to_join) == 2:
            return StridedInterval.pseudo_join(intervals_to_join[0], intervals_to_join[1])

        # sort the intervals in increasing left bound
        sorted_intervals = sorted(intervals_to_join, key=lambda x: x.lower_bound)
        # Fig 3 of the paper
        ret = None

        # we try all possible joins (linear with the number of SI to join)
        # and we return the one with the least number of values.
        for i in xrange(len(sorted_intervals)):
            # let's join all of them
            si = reduce(lambda x, y: StridedInterval.pseudo_join(x, y, False), sorted_intervals[i:] + sorted_intervals[0:i])

            if ret is None or ret.n_values > si.n_values:
                ret = si

        if any([x for x in intervals_to_join if x.uninitialized]):
            ret.uninitialized = True

        return ret