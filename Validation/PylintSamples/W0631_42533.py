def filter_from_url_arg(model_cls, query, arg, query_operator=and_,
                        arg_types=None):
    """
    Parse filter URL argument ``arg`` and apply to ``query``

    Example: 'column1<=value,column2==value' -> query.filter(Model.column1 <= value, Model.column2 == value)
    """

    fields = arg.split(',')
    mapper = class_mapper(model_cls)

    if not arg_types:
        arg_types = {}

    exprs = []
    joins = set()
    for expr in fields:
        if expr == "":
            continue

        e_mapper = mapper
        e_model_cls = model_cls

        operator = None
        method = None
        for op in operator_order:
            if op in expr:
                operator = op
                method = operator_to_method[op]
                break

        if operator is None:
            raise Exception('No operator in expression "{0}".'.format(expr))

        (column_names, value) = expr.split(operator)

        column_names = column_names.split('__')
        value = value.strip()

        for column_name in column_names:
            if column_name in arg_types:
                typed_value = arg_types[column_name](value)
            else:
                typed_value = value

            if column_name in e_mapper.relationships:
                joins.add(column_name)
                e_model_cls = e_mapper.attrs[column_name].mapper.class_
                e_mapper = class_mapper(e_model_cls)

        if hasattr(e_model_cls, column_name):
            column = getattr(e_model_cls, column_name)
            exprs.append(getattr(column, method)(typed_value))
        else:
            raise Exception('Invalid property {0} in class {1}.'.format(column_name, e_model_cls))

    return query.join(*joins).filter(query_operator(*exprs))