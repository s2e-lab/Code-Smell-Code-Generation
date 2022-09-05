def _create_typed_object_meta(get_fset):
    # type: (Callable[[str, str, Type[_T]], Callable[[_T], None]]) -> type
    """Create a metaclass for typed objects.

    Args:
        get_fset: A function that takes three parameters: the name of an
            attribute, the name of the private attribute that holds the
            property data, and a type. This function must an object method that
            accepts a value.

    Returns:
        A metaclass that reads annotations from a class definition and creates
        properties for annotated, public, non-constant, non-method attributes
        that will guarantee the type of the stored value matches the
        annotation.
    """

    def _get_fget(attr, private_attr, type_):
        # type: (str, str, Type[_T]) -> Callable[[], Any]
        """Create a property getter method for an attribute.

        Args:
            attr: The name of the attribute that will be retrieved.
            private_attr: The name of the attribute that will store any data
                related to the attribute.
            type_: The annotated type defining what values can be stored in the
                attribute.

        Returns:
            A function that takes self and retrieves the private attribute from
            self.
        """

        def _fget(self):
            # type: (...) -> Any
            """Get attribute from self without revealing the private name."""
            try:
                return getattr(self, private_attr)
            except AttributeError:
                raise AttributeError(
                    "'{}' object has no attribute '{}'".format(
                        _get_type_name(type_), attr
                    )
                )

        return _fget

    class _AnnotatedObjectMeta(type):
        """A metaclass that reads annotations from a class definition."""

        def __new__(
            mcs,  # type: Type[_AnnotatedObjectMeta]
            name,  # type: str
            bases,  # type: List[type]
            attrs,  # type: Dict[str, Any]
            **kwargs  # type: Dict[str, Any]
        ):
            # type: (...) -> type
            """Create class objs that replaces annotated attrs with properties.

            Args:
                mcs: The class object being created.
                name: The name of the class to create.
                bases: The list of all base classes for the new class.
                attrs: The list of all attributes for the new class from the
                    definition.

            Returns:
                A new class instance with the expected base classes and
                attributes, but with annotated, public, non-constant,
                non-method attributes replaced by property objects that
                validate against the annotated type.
            """
            annotations = attrs.get("__annotations__", {})
            use_comment_type_hints = (
                not annotations and attrs.get("__module__") != __name__
            )
            if use_comment_type_hints:
                frame_source = _get_class_frame_source(name)
                annotations = get_type_hints(*frame_source)
            names = list(attrs) + list(annotations)
            typed_attrs = {}
            for attr in names:
                typed_attrs[attr] = attrs.get(attr)
                if _is_propertyable(names, attrs, annotations, attr):
                    private_attr = "__{}".format(attr)
                    if attr in attrs:
                        typed_attrs[private_attr] = attrs[attr]
                    type_ = (
                        Optional[annotations[attr]]
                        if not use_comment_type_hints
                        and attr in attrs
                        and attrs[attr] is None
                        else annotations[attr]
                    )
                    typed_attrs[attr] = property(
                        _get_fget(attr, private_attr, type_),
                        get_fset(attr, private_attr, type_),
                    )
            properties = [
                attr
                for attr in annotations
                if _is_propertyable(names, attrs, annotations, attr)
            ]
            typed_attrs["_tp__typed_properties"] = properties
            typed_attrs["_tp__required_typed_properties"] = [
                attr
                for attr in properties
                if (
                    attr not in attrs
                    or attrs[attr] is None
                    and use_comment_type_hints
                )
                and NoneType not in getattr(annotations[attr], "__args__", ())
            ]
            return super(_AnnotatedObjectMeta, mcs).__new__(  # type: ignore
                mcs, name, bases, typed_attrs, **kwargs
            )

    return _AnnotatedObjectMeta