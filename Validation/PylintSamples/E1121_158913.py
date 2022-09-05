def reduce(self, f):
        """
        Reduces the elements of this RDD using the specified commutative and
        associative binary operator. Currently reduces partitions locally.

        >>> from operator import add
        >>> sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
        15
        >>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
        10
        >>> sc.parallelize([]).reduce(add)
        Traceback (most recent call last):
            ...
        ValueError: Can not reduce() empty RDD
        """
        f = fail_on_stopiteration(f)

        def func(iterator):
            iterator = iter(iterator)
            try:
                initial = next(iterator)
            except StopIteration:
                return
            yield reduce(f, iterator, initial)

        vals = self.mapPartitions(func).collect()
        if vals:
            return reduce(f, vals)
        raise ValueError("Can not reduce() empty RDD")