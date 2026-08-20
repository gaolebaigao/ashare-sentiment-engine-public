import os
import tempfile
import unittest
from pathlib import Path

from ashare_sentiment.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_expands_environment_default(self):
        config = load_config("config/default.yaml")
        self.assertEqual(config["data"]["provider"], os.environ.get("DATA_PROVIDER", "tushare"))
        self.assertEqual(sum(config["scoring"]["weights"].values()), 1.0)

    def test_overrides_are_deep_merged(self):
        config = load_config("config/default.yaml", {"data": {"exclude_st": False}})
        self.assertFalse(config["data"]["exclude_st"])
        self.assertEqual(config["data"]["min_periods"], 252)

    def test_invalid_weight_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text(
                "data: {provider: test}\nscoring: {weights: {breadth: 1}}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
