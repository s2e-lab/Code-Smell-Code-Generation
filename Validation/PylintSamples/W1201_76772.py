def adapt_persistent_to_rest(self, persistent_object, attribute_filter=None):
        """
        adapts a persistent model to a rest model by inspecting
        """
        # convert filter to immutable if it isn't already
        if isinstance(attribute_filter, parser.AttributeFilter):
            attribute_filter = attribute_filter.as_immutable()

        rest_model_instance = self.rest_model_class()

        for attribute_key in rest_model_instance.get_attribute_keys():

            # attribute is not visible don't bother processing
            if isinstance(attribute_filter, (parser.AttributeFilter, parser.AttributeFilterImmutable)) and \
               not attribute_filter.is_attribute_visible(attribute_key):
                continue

            rest_attr = getattr(self.rest_model_class, attribute_key)

            # don't bother processing if the persistent model doesn't have this attribute
            if not hasattr(persistent_object, attribute_key):

                if isinstance(rest_attr, types.Model):
                    #: If the attribute is a Model, then we set it to None otherwise we get a model
                    #: with default values, which is invalid when constructing responses
                    try:
                        setattr(rest_model_instance, attribute_key, None)
                    # catch any exception thrown from setattr to give a usable error message
                    except TypeError as exp:
                        raise TypeError('Attribute %s, %s' % (attribute_key, str(exp)))

                continue
            # ignore class methods
            elif inspect.ismethod(getattr(persistent_object, attribute_key)):
                import logging
                logging.error("ignoring method: "+attribute_key)
                continue

            # handles prestans array population from SQLAlchemy relationships
            elif isinstance(rest_attr, types.Array):

                persistent_attr_value = getattr(persistent_object, attribute_key)
                rest_model_array_handle = getattr(rest_model_instance, attribute_key)

                # iterator uses the .append method exposed by prestans arrays to validate
                # and populate the collection in the instance.
                for collection_element in persistent_attr_value:
                    if rest_attr.is_scalar:
                        rest_model_array_handle.append(collection_element)
                    else:
                        element_adapter = registry.get_adapter_for_rest_model(rest_attr.element_template)

                        # check if there is a sub model filter
                        sub_attribute_filter = None
                        if attribute_filter and attribute_key in attribute_filter:
                            sub_attribute_filter = getattr(attribute_filter, attribute_key)

                        adapted_rest_model = element_adapter.adapt_persistent_to_rest(
                            collection_element,
                            sub_attribute_filter
                        )
                        rest_model_array_handle.append(adapted_rest_model)

            elif isinstance(rest_attr, types.Model):

                try:
                    persistent_attr_value = getattr(persistent_object, attribute_key)

                    if persistent_attr_value is None:
                        adapted_rest_model = None
                    else:
                        model_adapter = registry.get_adapter_for_rest_model(rest_attr)

                        # check if there is a sub model filter
                        sub_attribute_filter = None
                        if isinstance(attribute_filter, (parser.AttributeFilter, parser.AttributeFilterImmutable)) and \
                                attribute_key in attribute_filter:
                            sub_attribute_filter = getattr(attribute_filter, attribute_key)

                        adapted_rest_model = model_adapter.adapt_persistent_to_rest(
                            persistent_attr_value,
                            sub_attribute_filter
                        )

                    setattr(rest_model_instance, attribute_key, adapted_rest_model)

                except TypeError as exp:
                    raise TypeError('Attribute %s, %s' % (attribute_key, str(exp)))
                except exception.DataValidationException as exp:
                    raise exception.InconsistentPersistentDataError(attribute_key, str(exp))

            else:

                # otherwise copy the value to the rest model
                try:
                    persistent_attr_value = getattr(persistent_object, attribute_key)
                    setattr(rest_model_instance, attribute_key, persistent_attr_value)
                except TypeError as exp:
                    raise TypeError('Attribute %s, %s' % (attribute_key, str(exp)))
                except exception.ValidationError as exp:
                    raise exception.InconsistentPersistentDataError(attribute_key, str(exp))

        return rest_model_instance