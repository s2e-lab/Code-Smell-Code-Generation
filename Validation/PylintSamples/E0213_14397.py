def decorate_class(self, klass, *decorator_args, **decorator_kwargs):
        """where the magic happens, this wraps a class to call our decorate method
        in the init of the class
        """
        class ChildClass(klass):
            def __init__(slf, *args, **kwargs):
                super(ChildClass, slf).__init__(*args, **kwargs)
                self.decorate(
                    slf, *decorator_args, **decorator_kwargs
                )

        decorate_klass = ChildClass
        decorate_klass.__name__ = klass.__name__
        decorate_klass.__module__ = klass.__module__
        # for some reason you can't update a __doc__ on a class
        # http://bugs.python.org/issue12773

        return decorate_klass