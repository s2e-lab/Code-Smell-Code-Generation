def _wrap_field(field):
    """Improve Flask-RESTFul's original field type"""
    class WrappedField(field):
        def output(self, key, obj):
            value = _fields.get_value(key if self.attribute is None else self.attribute, obj)

            # For all fields, when its value was null (None), return null directly,
            #  instead of return its default value (eg. int type's default value was 0)
            # Because sometimes the client **needs** to know, was a field of the model empty, to decide its behavior.
            return None if value is None else self.format(value)
    return WrappedField