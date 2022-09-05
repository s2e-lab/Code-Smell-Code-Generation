def permission_required(perm, *lookup_variables, **kwargs):
    """
    Decorator for views that checks whether a user has a particular permission
    enabled, redirecting to the log-in page if necessary.
    """
    login_url = kwargs.pop('login_url', settings.LOGIN_URL)
    redirect_field_name = kwargs.pop('redirect_field_name', REDIRECT_FIELD_NAME)
    redirect_to_login = kwargs.pop('redirect_to_login', True)

    def decorate(view_func):
        def decorated(request, *args, **kwargs):
            if request.user.is_authenticated():
                params = []
                for lookup_variable in lookup_variables:
                    if isinstance(lookup_variable, string_types):
                        value = kwargs.get(lookup_variable, None)
                        if value is None:
                            continue
                        params.append(value)
                    elif isinstance(lookup_variable, (tuple, list)):
                        model, lookup, varname = lookup_variable
                        value = kwargs.get(varname, None)
                        if value is None:
                            continue
                        if isinstance(model, string_types):
                            model_class = apps.get_model(*model.split("."))
                        else:
                            model_class = model
                        if model_class is None:
                            raise ValueError(
                                "The given argument '%s' is not a valid model." % model)
                        if (inspect.isclass(model_class) and
                                not issubclass(model_class, Model)):
                            raise ValueError(
                                'The argument %s needs to be a model.' % model)
                        obj = get_object_or_404(model_class, **{lookup: value})
                        params.append(obj)
                check = get_check(request.user, perm)
                granted = False
                if check is not None:
                    granted = check(*params)
                if granted or request.user.has_perm(perm):
                    return view_func(request, *args, **kwargs)
            if redirect_to_login:
                path = urlquote(request.get_full_path())
                tup = login_url, redirect_field_name, path
                return HttpResponseRedirect('%s?%s=%s' % tup)
            return permission_denied(request)
        return wraps(view_func)(decorated)
    return decorate