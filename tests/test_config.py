import random
import unittest

import config


class ConfigTests(unittest.TestCase):
    def test_seed_reproducibility_for_random_module(self):
        config.set_global_seed(42)
        first = [random.random() for _ in range(3)]
        config.set_global_seed(42)
        second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)

    def test_expected_seed_constant(self):
        self.assertEqual(config.SEED, 42)

    def test_live_key_detection(self):
        self.assertFalse(config.has_live_api_key("your_key_here"))
        self.assertTrue(config.has_live_api_key("real-key"))


if __name__ == "__main__":
    unittest.main()
