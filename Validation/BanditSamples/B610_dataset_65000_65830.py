def _perform_unique_localized_field_checks(self, unique_checks):
        """
        Do the checks for the localized fields.
        """
        bad_fields = set()
        form_errors = []

        for (field_name, local_field_name) in unique_checks:
            
            lookup_kwargs = {}
            lookup_value = self.cleaned_data[field_name]
            # ModelChoiceField will return an object instance rather than
            # a raw primary key value, so convert it to a pk value before
            # using it in a lookup.
            lookup_value = getattr(lookup_value, 'pk', lookup_value)
            lookup_kwargs[str(local_field_name)] = lookup_value

            qs = self.instance.__class__._default_manager.filter(**lookup_kwargs)

            # Exclude the current object from the query if we are editing an
            # instance (as opposed to creating a new one)
            if self.instance.pk is not None:
                qs = qs.exclude(pk=self.instance.pk)

            # This cute trick with extra/values is the most efficient way to
            # tell if a particular query returns any results.
            if qs.extra(select={'a': 1}).values('a').order_by():
                self._errors[field_name] = ErrorList([self.unique_error_message([field_name])])
                bad_fields.add(field_name)
                
        return bad_fields, form_errors