import logging


from MTNeuro.self_supervised.utils import you_only_log_once

log = logging.getLogger(__name__)
yolo = you_only_log_once(traceback=1)  # the caller only uses this once


def pass_on_failure(f):
    r"""Doesn't raise error."""
    def applicator(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            with yolo as go_ahead:
                log.warning('Call to function %r failed: %r.', f.__name__, e) if go_ahead else None
    return applicator
