def metered_meta(metrics, base=type):
    """Creates a metaclass that will add the specified metrics at a path parametrized on the dynamic class name.

    Prime use case is for base classes if all subclasses need separate metrics and / or the metrics need to be
    used in base class methods, e.g., Tornado's ``RequestHandler`` like::

        import tapes
        import tornado
        import abc

        registry = tapes.Registry()

        class MyCommonBaseHandler(tornado.web.RequestHandler):
            __metaclass__ = metered_meta([
                ('latency', 'my.http.endpoints.{}.latency', registry.timer)
            ], base=abc.ABCMeta)

            @tornado.gen.coroutine
            def get(self, *args, **kwargs):
                with self.latency.time():
                    yield self.get_impl(*args, **kwargs)

            @abc.abstractmethod
            def get_impl(self, *args, **kwargs):
                pass


        class MyImplHandler(MyCommonBaseHandler):
            @tornado.gen.coroutine
            def get_impl(self, *args, **kwargs):
                self.finish({'stuff': 'something'})


        class MyOtherImplHandler(MyCommonBaseHandler):
            @tornado.gen.coroutine
            def get_impl(self, *args, **kwargs):
                self.finish({'other stuff': 'more of something'})

    This would produce two different relevant metrics,
        - ``my.http.endpoints.MyImplHandler.latency``
        - ``my.http.endpoints.MyOtherImplHandler.latency``

    and, as an unfortunate side effect of adding it in the base class,
    a ``my.http.endpoints.MyCommonBaseHandler.latency`` too.

    :param metrics: list of (attr_name, metrics_path_template, metrics_factory)
    :param base: optional meta base if other than `type`
    :return: a metaclass that populates the class with the needed metrics at paths based on the dynamic class name
    """
    class _MeteredMeta(base):
        def __new__(meta, name, bases, dict_):
            new_dict = dict(**dict_)
            for attr_name, template, factory in metrics:
                new_dict[attr_name] = factory(template.format(name))
            return super(_MeteredMeta, meta).__new__(meta, name, bases, new_dict)

    return _MeteredMeta