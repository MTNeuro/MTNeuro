from inspect import currentframe


class you_only_log_once:
    def __init__(self, traceback=0):
        self.traceback = traceback
        self.enter_flag = True
        self.lines = []

    def __enter__(self):
        if self.enter_flag:
            cf = currentframe()
            caller = cf.f_back
            for _ in range(self.traceback):
                caller = caller.f_back
            line_no = cf.f_back.f_lineno
            if line_no not in self.lines:
                self.lines.append(line_no)
                return True
        return False

    def __call__(self, enter_flag=True):
        self.enter_flag = enter_flag
        return self

    def __exit__(self, *args):
        self.enter_flag = True
        return
