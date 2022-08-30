class KwargsParser():
    '''
    Class to handle the kwargs parsing.
    '''

    def __init__(self, kwargs, name=None):
        '''
        Class initializer

        Parameters
        ----------
        kwargs : dict
            Input kwargs
        name : str or None, default: None
            Used when printing warnings to specify instance
        '''
        self.kwargs = kwargs
        self.name = name

    def get_safe(self, key, default_val, type=None, verbose=False):
        '''
        Get the value with the specified key casted to a specific type if requested.
        If the key is not present in the kwargs dict the default value is returned. 

        Parameters
        ----------
        key : str
            Input key
        default_val : type
            Default value if the key is not present in the kwargs
        type : dtype or None, default: None
            Requested type, if None no type casting is executed
        verbose : bool, default: False
            If True a warning is printed for missing keys if name is not None

        Returns
        -------
        val : dtype
            Parsed value
        '''
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
                    print(
                        self.name + ': ' +
                        key + ' not present in kwargs, using default value:',
                        type(default_val)
                        )
            return type(default_val)
