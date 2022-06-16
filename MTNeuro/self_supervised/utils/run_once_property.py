from functools import wraps


def run_once_property(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            instance = getattr(self, '_' + fn.__name__)
            return instance
        except AttributeError:
            instance = fn(self, *args, **kwargs)
            setattr(self, '_' + fn.__name__, instance)
            return instance
    return property(wrapper)