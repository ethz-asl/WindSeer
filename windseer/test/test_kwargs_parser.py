#!/usr/bin/env python
'''
Testcases for the KwargsParser
'''

from windseer.utils import KwargsParser

import unittest


class TestKwargsParser(unittest.TestCase):

    def run_case(self, type, default_value, use_name=False):
        kwargs = {
            bool.__name__: True,
            int.__name__: 1,
            float.__name__: 1.0,
            str.__name__: 'abc',
            list.__name__: [0, 1],
            dict.__name__: {
                'a': 1,
                'b': 2
                }
            }

        if use_name:
            parser = KwargsParser(kwargs, 'Test')
        else:
            parser = KwargsParser(kwargs)

        val = parser.get_safe(type.__name__, default_value, type, True)
        val_default = parser.get_safe('invalid_key', default_value, type, True)

        self.assertEqual(val, kwargs[type.__name__])
        self.assertEqual(val_default, default_value)

    def test_get_bool(self):
        self.run_case(bool, False)

    def test_get_bool_named(self):
        self.run_case(bool, False, True)

    def test_get_int(self):
        self.run_case(int, 50)

    def test_get_float(self):
        self.run_case(float, 1.23456789)

    def test_get_str(self):
        self.run_case(str, 'def')

    def test_get_list(self):
        self.run_case(list, [4, 5, 6])

    def test_get_dict(self):
        self.run_case(dict, {'c': 3, 'd': 4})


if __name__ == '__main__':
    unittest.main()
