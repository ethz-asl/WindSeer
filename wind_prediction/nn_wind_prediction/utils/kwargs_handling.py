class KwargsParser():
    def __init__(self, kwargs, name = None):
        self.kwargs = kwargs
        self.name = name

    def get_safe(self, key, default_val, type = None, verbose=False):
        try:
            if isinstance(self.kwargs[key], str):
                val = self.kwargs[key].replace(',', '')
            else:
                val = self.kwargs[key]

            if type is not None:
                return type(val)
            else:
                return val

        except KeyError:
            if verbose:
                if self.name is not None:
                    print(self.name + ': ' + key + ' not present in kwargs, using default value:', type(default_val))
            return type(default_val)
