"""
Factory for creating configuration objects based on environment.
"""

from enum import Enum
from typing import Union

from .base_config import BaseConfig
from .min_config import MinConfig
from .dev_config import DevConfig
from .staging_config import StagingConfig
from .prod_config import ProdConfig

class Environment(Enum):
    MIN = "min"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

class ConfigFactory:
    @staticmethod
    def get_config(env: Union[str, Environment]) -> BaseConfig:
        """
        Create and return appropriate configuration based on environment.

        Args:
            env: Environment name or Environment enum

        Returns:
            Configuration object for specified environment

        Raises:
            ValueError: If invalid environment specified
        """
        if isinstance(env, str):
            env = Environment(env.lower())

        config_map = {
            Environment.MIN: MinConfig,
            Environment.DEV: DevConfig,
            Environment.STAGING: StagingConfig,
            Environment.PROD: ProdConfig
        }

        if env not in config_map:
            raise ValueError(f"Invalid environment: {env}")

        return config_map[env]()
