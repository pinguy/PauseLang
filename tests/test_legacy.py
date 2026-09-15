"""Expose every original torture test independently to unittest and CI."""
import random
import unittest
from tests.legacy_torture import TortureTests


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.random_state = random.getstate()
        random.seed(713)

    def tearDown(self):
        random.setstate(self.random_state)


def wrap(fn):
    def test(self):
        fn()
    return test


for name in dir(TortureTests):
    if name.startswith('test_'):
        setattr(LegacyTests, name, wrap(getattr(TortureTests, name)))
