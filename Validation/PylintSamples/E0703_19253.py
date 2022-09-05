def replace_exceptions(
    old_to_new_exceptions: Dict[Type[BaseException], Type[BaseException]]
) -> Callable[..., Any]:
    """
    Replaces old exceptions with new exceptions to be raised in their place.
    """
    old_exceptions = tuple(old_to_new_exceptions.keys())

    def decorator(to_wrap: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(to_wrap)
        # String type b/c pypy3 throws SegmentationFault with Iterable as arg on nested fn
        # Ignore so we don't have to import `Iterable`
        def wrapper(
            *args: Iterable[Any], **kwargs: Dict[str, Any]
        ) -> Callable[..., Any]:
            try:
                return to_wrap(*args, **kwargs)
            except old_exceptions as err:
                try:
                    raise old_to_new_exceptions[type(err)] from err
                except KeyError:
                    raise TypeError(
                        "could not look up new exception to use for %r" % err
                    ) from err

        return wrapper

    return decorator