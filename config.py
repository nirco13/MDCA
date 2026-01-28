"""
Configuration loader.

Supports YAML config files for reproducible experiments.
"""

import yaml
from argparse import Namespace
import os


def load_config(config_path: str) -> Namespace:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        args: Namespace object with all configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Flatten nested dict to Namespace
    flat_dict = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            flat_dict.update(values)
        else:
            flat_dict[section] = values

    # Ensure device is set
    if 'device' not in flat_dict:
        import torch
        flat_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return Namespace(**flat_dict)


def save_config(args: Namespace, save_path: str):
    """
    Save configuration to YAML file for reproducibility.

    Args:
        args: Configuration namespace
        save_path: Path to save YAML file
    """
    config_dict = vars(args)

    # Convert non-serializable objects
    serializable_dict = {}
    for key, value in config_dict.items():
        if hasattr(value, '__dict__'):
            serializable_dict[key] = str(value)
        else:
            serializable_dict[key] = value

    with open(save_path, 'w') as f:
        yaml.dump(serializable_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {save_path}")
