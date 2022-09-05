def has(self, relation, operator=">=", count=1, boolean="and", extra=None):
        """
        Add a relationship count condition to the query.

        :param relation: The relation to count
        :type relation: str

        :param operator: The operator
        :type operator: str

        :param count: The count
        :type count: int

        :param boolean: The boolean value
        :type boolean: str

        :param extra: The extra query
        :type extra: Builder or callable

        :type: Builder
        """
        if relation.find(".") >= 0:
            return self._has_nested(relation, operator, count, boolean, extra)

        relation = self._get_has_relation_query(relation)

        query = relation.get_relation_count_query(
            relation.get_related().new_query(), self
        )

        # TODO: extra query
        if extra:
            if callable(extra):
                extra(query)

        return self._add_has_where(
            query.apply_scopes(), relation, operator, count, boolean
        )