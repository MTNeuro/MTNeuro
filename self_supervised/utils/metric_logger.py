from functools import wraps

# todo add unittest

def consistentmethod(fn):
    @wraps(fn)
    def wrapper(self, **kwargs):
        attr = '_' + fn.__name__ + 'consistency_table'  # private
        if not hasattr(self, attr):
            setattr(self, attr, set(kwargs.keys()))
        else:
            if getattr(self, attr) != set(kwargs.keys()):
                missing = getattr(self, attr) - set(kwargs.keys())
                extra = set(kwargs.keys()) - getattr(self, attr)
                raise ValueError('Arguments are not historically consistent: Missing: %r. Extra: %r.' % (missing, extra))
        return fn(self, **kwargs)
    return wrapper


class MetricLogger:
    r"""Keeps track of training and validation curves, by recording:
        - Last value of train and validation metrics.
        - Train and validation metrics corresponding to maximum or minimum validation metric value.
        - Exponential moving average of train and validation metrics.

    Args:
        smoothing_factor (float, Optional): Smoothing factor used in exponential moving average.
            (default: :obj:`0.4`).
        max (bool, Optional): If :obj:`True`, tracks max value. Otherwise, tracks min value. (default: :obj:`True`).
    """
    def __init__(self, moving_average=False, smoothing_factor=None, early_stopping=False, max=None):
        # history
        self.train_hist = None
        self.val_hist = None
        self.test_hist = None

        # last
        self.train_last = None
        self.val_last = None
        self.test_last = None

        self.moving_average = moving_average
        if self.moving_average:
            assert smoothing_factor is not None and (0. < smoothing_factor < 1.)
            self.smoothing_factor = smoothing_factor
            self.train_smooth = None
            self.val_smooth = None
            self.test_smooth = None

        self.early_stopping = early_stopping
        if self.early_stopping:
            assert max is not None and isinstance(max, bool)
            self.max = max
            self.train_minmax = None
            self.val_minmax = None
            self.test_minmax = None
            self.step_total = None
            self.step_minmax = None

    @consistentmethod
    def update(self, *, train, val=None, test=None, step=None):
        # todo add check for arguments being consistent
        # update history
        if self.train_hist is None:
            self.train_hist = {step: train} if step is not None else [train]
            self.val_hist = {step: val} if step is not None else [val] if val is not None else None
            self.test_hist = {step: test} if step is not None else [test] if test is not None else None
        else:
            if step is not None:
                self.train_hist[step] = train
                self.val_hist[step] = val if val is not None else None
                self.test_hist[step] = test if test is not None else None
            else:
                self.train_hist.append(train)
                self.val_hist.append(val) if val is not None else None
                self.test_hist.append(test) if test is not None else None

        # last values
        self.train_last = train
        self.val_last = val
        self.test_last = test

        # exponential moving average
        if self.moving_average:
            self.train_smooth = self.smoothing_factor * train + (1 - self.smoothing_factor) * self.train_smooth \
                if (self.train_smooth is not None and train is not None) else train
            self.val_smooth = self.smoothing_factor * val + (1 - self.smoothing_factor) * self.val_smooth \
                if (self.val_smooth is not None and val is not None) else val
            self.test_smooth = self.smoothing_factor * test + (1 - self.smoothing_factor) * self.test_smooth \
                if (self.test_smooth is not None and test is not None) else test

        # max/min validation accuracy
        if self.early_stopping:
            if val is None:
                raise ValueError("Validation isn't being tracked.")
            if self.val_minmax is None or (self.max and self.val_minmax < val) or \
                    (not self.max and self.val_minmax > val):
                self.train_minmax = train
                self.val_minmax = val
                self.test_minmax = test
                self.step_minmax = step
            self.step_total = step

    def __getattr__(self, item):
        if item not in ['train_min', 'train_max', 'val_min', 'val_max', 'test_min', 'test_max']:
            raise AttributeError
        if self.max and item in ['train_min', 'val_min', 'test_min']:
            raise AttributeError('Tracking maximum values, not minimum.')
        if not self.max and item in ['train_max', 'val_max', 'test_max']:
            raise AttributeError('Tracking minimum values, not maximum.')

        if 'train' in item:
            return self.train_minmax
        elif 'val' in item:
            return self.val_minmax
        elif 'test' in item:
            return self.test_minmax

    def __repr__(self):
        rep = lambda vals: ' '.join(["(%s) %.4f" % (key, val) for key, val in zip(['Train', 'Val', 'Test'], vals)
                                     if val is not None])

        out = "Accuracy       : " + rep((self.train_last, self.val_last, self.test_last))
        if self.moving_average:
            out += "\nSmooth Accuracy: " + rep((self.train_smooth, self.val_smooth, self.test_smooth))
        if self.early_stopping:
            step_str = " (Step) %d/%d" % (self.step_minmax, self.step_total) if self.step_minmax is not None else "NaN"
            out += "\nEarly stopping : " + rep((self.train_minmax, self.val_minmax, self.test_minmax)) + step_str
        return out

    def hist(self, step):
        # todo assumes all 3 are defined
        return self.train_hist[step], self.val_hist[step], self.test_hist[step]