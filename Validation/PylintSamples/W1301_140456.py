def _expand_prefix_query(self, query, table_name = None):
        """ Expand prefix query dict into a WHERE-clause.

            If you need to prefix each column reference with a table
            name, that can be supplied via the table_name argument.
        """

        where = unicode()
        opt = list()

        # handle table name, can be None
        if table_name is None:
            col_prefix = ""
        else:
            col_prefix = table_name + "."

        if 'val1' not in query:
            raise NipapMissingInputError("'val1' must be specified")
        if 'val2' not in query:
            raise NipapMissingInputError("'val2' must be specified")

        if type(query['val1']) == dict and type(query['val2']) == dict:
            # Sub expression, recurse! This is used for boolean operators: AND OR
            # add parantheses

            sub_where1, opt1 = self._expand_prefix_query(query['val1'], table_name)
            sub_where2, opt2 = self._expand_prefix_query(query['val2'], table_name)
            try:
                where += unicode(" (%s %s %s) " % (sub_where1, _operation_map[query['operator']], sub_where2) )
            except KeyError:
                raise NipapNoSuchOperatorError("No such operator %s" % unicode(query['operator']))

            opt += opt1
            opt += opt2

        else:

            # TODO: raise exception if someone passes one dict and one "something else"?

            # val1 is key, val2 is value.

            if query['val1'] not in _prefix_spec:
                raise NipapInputError('Search variable \'%s\' unknown' % unicode(query['val1']))

            # build where clause
            if query['operator'] not in _operation_map:
                raise NipapNoSuchOperatorError("No such operator %s" % query['operator'])

            if query['val1'] == 'vrf_id' and query['val2'] is None:
                query['val2'] = 0

            # workaround for handling equal matches of NULL-values
            if query['operator'] == 'equals' and query['val2'] is None:
                query['operator'] = 'is'
            elif query['operator'] == 'not_equals' and query['val2'] is None:
                query['operator'] = 'is_not'

            if query['operator'] in (
                    'contains',
                    'contains_equals',
                    'contained_within',
                    'contained_within_equals'):

                where = " iprange(prefix) %(operator)s %%s " % {
                        'col_prefix': col_prefix,
                        'operator': _operation_map[query['operator']]
                        }

            elif query['operator'] in ('equals_any',):
                where = unicode(" %%s = ANY (%s%s::citext[]) " %
                        ( col_prefix, _prefix_spec[query['val1']]['column'])
                        )

            elif query['operator'] in (
                    'like',
                    'regex_match',
                    'regex_not_match'):
                # we COALESCE column with '' to allow for example a regexp
                # search on '.*' to match columns which are NULL in the
                # database
                where = unicode(" COALESCE(%s%s, '') %s %%s " %
                        ( col_prefix, _prefix_spec[query['val1']]['column'],
                        _operation_map[query['operator']] )
                        )

            else:
                where = unicode(" %s%s %s %%s " %
                        ( col_prefix, _prefix_spec[query['val1']]['column'],
                        _operation_map[query['operator']] )
                        )

            opt.append(query['val2'])

        return where, opt