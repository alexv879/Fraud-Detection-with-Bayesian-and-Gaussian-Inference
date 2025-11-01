"""
Configuration Management Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module handles loading and validation of application configuration
from config.json. It provides centralized access to all settings used
throughout the Private AI File Organiser.

NOTICE: This software is proprietary and confidential. Unauthorized copying,
modification, distribution, or use is strictly prohibited.
See LICENSE.txt for full terms and conditions.

Version: 1.0.0
Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """
    Configuration manager that loads and provides access to application settings.

    Attributes:
        config_path (Path): Path to the config.json file
        _config (Dict[str, Any]): Loaded configuration dictionary
    """

    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.

        Args:
            config_path (str, optional): Path to config file. Defaults to project root config.json
        """
        if config_path is None:
            # Default to config.json in project root
            self.config_path = Path(__file__).parent.parent / "config.json"
        else:
            self.config_path = Path(config_path)

        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        Load configuration from JSON file with validation.

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is malformed
            ValueError: If required configuration keys are missing (MEDIUM #1 FIX)
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)

        # Validate required configuration keys (MEDIUM #1 FIX - Robustness)
        required_keys = ['watched_folders', 'ollama_model', 'base_destination']
        missing_keys = [key for key in required_keys if key not in self._config]

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}. "
                f"Please add these to {self.config_path}"
            )

        # Validate watched_folders is a list
        if not isinstance(self._config.get('watched_folders'), list):
            raise ValueError("'watched_folders' must be a list of directory paths")

        # Validate ollama_model is a string
        if not isinstance(self._config.get('ollama_model'), str):
            raise ValueError("'ollama_model' must be a string")

        # Expand home directory paths
        if "watched_folders" in self._config:
            self._config["watched_folders"] = [
                os.path.expanduser(path) for path in self._config["watched_folders"]
            ]
        # Expand base_destination if provided
        if "base_destination" in self._config:
            self._config["base_destination"] = os.path.expanduser(self._config["base_destination"])

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key (str): Configuration key (supports nested keys with dot notation)
            default (Any, optional): Default value if key not found

        Returns:
            Any: Configuration value or default

        Example:
            >>> config.get("ollama_model")
            'llama3'
            >>> config.get("time_estimates.move")
            0.5
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def watched_folders(self) -> List[str]:
        """Get list of folders to watch for new files."""
        return self.get("watched_folders", [])

    @property
    def destination_rules(self) -> Dict[str, str]:
        """Get file extension to destination path mapping."""
        return self.get("destination_rules", {})

    @property
    def ollama_model(self) -> str:
        """Get Ollama model name."""
        return self.get("ollama_model", "llama3")

    @property
    def ollama_base_url(self) -> str:
        """Get Ollama API base URL."""
        return self.get("ollama_base_url", "http://localhost:11434")

    @property
    def auto_mode(self) -> bool:
        """Check if automatic file processing is enabled."""
        return self.get("auto_mode", False)

    @property
    def dry_run(self) -> bool:
        """Check if dry run mode is enabled (no actual file operations)."""
        return self.get("dry_run", True)

    @property
    def time_estimates(self) -> Dict[str, float]:
        """Get time estimates (in minutes) for different operations."""
        return self.get("time_estimates", {
            "move": 0.5,
            "rename": 0.3,
            "delete": 0.2,
            "archive": 0.4
        })

    @property
    def enable_ai(self) -> bool:
        """Check if AI classification is enabled."""
        return self.get("classification.enable_ai", True)

    @property
    def text_extract_limit(self) -> int:
        """Get maximum characters to extract from files for AI analysis."""
        return self.get("classification.text_extract_limit", 500)

    @property
    def hash_algorithm(self) -> str:
        """Get hash algorithm for duplicate detection."""
        return self.get("duplicates.hash_algorithm", "sha1")

    @property
    def license_api_endpoint(self) -> str:
        """Get license verification API endpoint."""
        return self.get("license.api_endpoint", "")

    @property
    def license_offline_mode(self) -> bool:
        """Check if offline license validation is enabled."""
        return self.get("license.offline_mode", True)

    @property
    def base_destination(self) -> str:
        """Get base destination directory for suggested paths.

        By default this is the user's home directory but can be overridden in config.json
        with the `base_destination` key. The value will be expanded (e.g. ~ -> home).
        """
        return self.get("base_destination", str(Path.home()))

    @property
    def path_blacklist(self) -> List[str]:
        """List of paths or path prefixes that must not be processed or moved."""
        return self.get("path_blacklist", [])

    @property
    def folder_policies(self) -> Dict[str, Any]:
        """Per-folder policy overrides. Example structure:

        {
            "C:/Users/Alice/Downloads": {"auto_mode": false, "allow_move": false},
            "D:/Incoming": {"auto_mode": true}
        }
        """
        return self.get("folder_policies", {})

    def save(self) -> None:
        """
        Save current configuration back to JSON file.
        Useful for updating settings programmatically.
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)

    def update(self, key: str, value: Any) -> None:
        """
        Update configuration value.

        Args:
            key (str): Configuration key (supports nested keys with dot notation)
            value (Any): New value to set

        Example:
            >>> config.update("auto_mode", True)
            >>> config.update("time_estimates.move", 0.7)
        """
        keys = key.split('.')
        target = self._config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def _is_path_blacklisted(self, path: Path, blacklist: List[str]) -> bool:
        """
        Check if path (or symlink target) is blacklisted (HIGH #7 FIX).

        This method checks both the symlink location and its target to prevent
        bypassing blacklists via symlinks.

        Args:
            path (Path): Path to check
            blacklist (List[str]): List of blacklisted paths

        Returns:
            bool: True if path is blacklisted, False otherwise
        """
        try:
            if path.is_symlink():
                # Check both symlink location and target
                symlink_path = path
                target_path = path.resolve()

                for check_path in [symlink_path, target_path]:
                    resolved = str(check_path)
                    for blacklisted in blacklist:
                        blacklisted_resolved = str(Path(blacklisted).expanduser().resolve())
                        try:
                            if os.path.commonpath([resolved, blacklisted_resolved]) == blacklisted_resolved:
                                return True
                        except (ValueError, OSError):
                            if resolved.lower().startswith(blacklisted_resolved.lower()):
                                return True
            else:
                # Normal path check
                resolved = str(path.resolve())
                for blacklisted in blacklist:
                    blacklisted_resolved = str(Path(blacklisted).expanduser().resolve())
                    try:
                        if os.path.commonpath([resolved, blacklisted_resolved]) == blacklisted_resolved:
                            return True
                    except (ValueError, OSError):
                        if resolved.lower().startswith(blacklisted_resolved.lower()):
                            return True

        except Exception:
            return True  # If check fails, err on side of caution

        return False

    def get_folder_policy(self, path: str) -> Dict[str, Any]:
        """
        Get the most specific folder policy for a given path.

        This method finds the deepest (most specific) ancestor policy that
        applies to the given path. For example, if policies exist for both
        "C:/Users" and "C:/Users/Alice/Downloads", and the path is
        "C:/Users/Alice/Downloads/file.txt", the Downloads policy is returned.

        Args:
            path (str): Path to check for policies

        Returns:
            Dict or None: Folder policy dict with keys like:
                - auto_mode (bool): Override auto_mode for this folder
                - allow_move (bool): Whether moves are allowed
                - use_ai (bool): Whether to use AI for this folder
                Returns None if no policy matches.

        Example:
            >>> policy = config.get_folder_policy("/home/user/Downloads/file.txt")
            >>> if policy and not policy.get('allow_move'):
            ...     print("Moves disabled for this folder")
        """
        folder_policies = self.folder_policies
        if not folder_policies:
            return None

        try:
            # Resolve the target path
            target_path = Path(path).resolve()

            # Find all matching policies (where policy path is ancestor of target)
            matching_policies = []

            for policy_key, policy_value in folder_policies.items():
                try:
                    # Expand and resolve policy path
                    policy_path = Path(policy_key).expanduser().resolve()

                    # Check if policy_path is an ancestor of target_path
                    try:
                        # For same-drive paths, use commonpath
                        if os.path.commonpath([str(target_path), str(policy_path)]) == str(policy_path):
                            # Calculate depth (deeper = more specific)
                            depth = len(policy_path.parts)
                            matching_policies.append((depth, policy_value))
                    except ValueError:
                        # Different drives on Windows - check prefix instead
                        if str(target_path).lower().startswith(str(policy_path).lower()):
                            depth = len(policy_path.parts)
                            matching_policies.append((depth, policy_value))

                except Exception:
                    # If path resolution fails for this policy, skip it
                    continue

            # Return the deepest (most specific) matching policy
            if matching_policies:
                matching_policies.sort(key=lambda x: x[0], reverse=True)
                return matching_policies[0][1]

            return None

        except Exception:
            # If any error occurs, return None (no policy)
            return None


# Global configuration instance
_config_instance: Config = None


def get_config(config_path: str = None) -> Config:
    """
    Get or create global configuration instance.

    Args:
        config_path (str, optional): Path to config file

    Returns:
        Config: Global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Watched folders: {config.watched_folders}")
    print(f"Ollama model: {config.ollama_model}")
    print(f"Auto mode: {config.auto_mode}")
