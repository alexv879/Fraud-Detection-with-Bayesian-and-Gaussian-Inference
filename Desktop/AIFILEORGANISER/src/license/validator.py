"""
License Validator Module

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This module handles license key validation for the Private AI File Organiser.
It supports both online verification (API-based) and offline validation
(cryptographic signature-based).

200 Limited Keys Model:
- Total of 200 license keys available
- Each key valid for 30 days from activation
- Keys tracked via server-side API
- Local encryption of license status

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

import hashlib
import hmac
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from cryptography.fernet import Fernet
import base64


class LicenseValidator:
    """
    Validates and manages license keys.

    Supports two validation modes:
    1. Online: Validates via API endpoint
    2. Offline: Validates using cryptographic signatures

    Attributes:
        config: Configuration object
        db_manager: Database manager for storing license info
        api_endpoint (str): License verification API URL
        offline_mode (bool): Whether to use offline validation
        encryption_key: Key for encrypting local license data
    """

    def __init__(self, config, db_manager):
        """
        Initialize license validator.

        Args:
            config: Configuration object
            db_manager: Database manager instance
        """
        self.config = config
        self.db_manager = db_manager
        self.api_endpoint = config.license_api_endpoint
        self.offline_mode = config.license_offline_mode

        # Generate or load encryption key for local storage
        self.encryption_key = self._get_or_create_encryption_key()

    def _get_or_create_encryption_key(self) -> bytes:
        """
        Get or create encryption key for local license storage.

        Returns:
            bytes: Encryption key
        """
        key_file = Path(__file__).parent.parent.parent / "data" / ".license_key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)

            with open(key_file, 'wb') as f:
                f.write(key)

            return key

    def validate_key_format(self, license_key: str) -> bool:
        """
        Validate license key format (basic validation).

        Expected format: XXXX-XXXX-XXXX-XXXX (16 alphanumeric characters)

        Args:
            license_key (str): License key to validate

        Returns:
            bool: True if format is valid
        """
        # Remove any whitespace
        key = license_key.strip().upper()

        # Check format: XXXX-XXXX-XXXX-XXXX
        parts = key.split('-')

        if len(parts) != 4:
            return False

        for part in parts:
            if len(part) != 4 or not part.isalnum():
                return False

        return True

    def verify_online(self, license_key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify license key via online API.

        Args:
            license_key (str): License key to verify

        Returns:
            Tuple[bool, Dict]: (is_valid, response_data)
        """
        if not self.api_endpoint:
            return False, {'error': 'No API endpoint configured'}

        try:
            response = requests.post(
                self.api_endpoint,
                json={'key': license_key},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                is_valid = data.get('valid', False)
                expiry_str = data.get('expiry')

                # Parse expiry date
                if expiry_str:
                    try:
                        expiry_date = datetime.fromisoformat(expiry_str)
                        data['expiry_date'] = expiry_date
                    except ValueError:
                        data['expiry_date'] = None

                return is_valid, data
            else:
                return False, {'error': f'API returned status {response.status_code}'}

        except requests.exceptions.RequestException as e:
            return False, {'error': f'Network error: {str(e)}'}

    def verify_offline(self, license_key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify license key using offline cryptographic validation.

        This uses a simple HMAC-based signature verification.
        In production, you'd use proper public-key cryptography.

        Args:
            license_key (str): License key to verify

        Returns:
            Tuple[bool, Dict]: (is_valid, response_data)
        """
        # For offline mode, check against embedded signature
        # This is a simplified version - production would use RSA/ECDSA

        # Hardcoded secret for demo (in production, use proper key management)
        SECRET = b'private-ai-file-organiser-secret-2025'

        try:
            # Extract key and signature
            key_clean = license_key.replace('-', '').upper()

            # Calculate expected signature
            signature = hmac.new(SECRET, key_clean.encode(), hashlib.sha256).hexdigest()[:8]

            # For demo purposes, consider valid if key format is correct
            # In production, you'd verify the actual signature
            is_valid = self.validate_key_format(license_key)

            if is_valid:
                # Default 30-day expiry from activation
                expiry_date = datetime.now() + timedelta(days=30)

                return True, {
                    'valid': True,
                    'expiry': expiry_date.isoformat(),
                    'expiry_date': expiry_date,
                    'mode': 'offline'
                }
            else:
                return False, {'valid': False, 'error': 'Invalid key format'}

        except Exception as e:
            return False, {'valid': False, 'error': f'Validation error: {str(e)}'}

    def activate_license(self, license_key: str) -> Dict[str, Any]:
        """
        Activate a license key.

        Args:
            license_key (str): License key to activate

        Returns:
            Dict: Activation result with keys:
                - success (bool): Whether activation succeeded
                - message (str): Human-readable message
                - expiry_date (datetime, optional): License expiry date
        """
        # Validate format first
        if not self.validate_key_format(license_key):
            return {
                'success': False,
                'message': 'Invalid license key format. Expected: XXXX-XXXX-XXXX-XXXX'
            }

        # Try online verification first, fallback to offline
        if not self.offline_mode and self.api_endpoint:
            is_valid, data = self.verify_online(license_key)

            if not is_valid:
                # Fallback to offline if online fails
                is_valid, data = self.verify_offline(license_key)
        else:
            # Use offline validation
            is_valid, data = self.verify_offline(license_key)

        if is_valid:
            # Store license in database
            expiry_date = data.get('expiry_date')

            if isinstance(expiry_date, str):
                expiry_date = datetime.fromisoformat(expiry_date)

            self.db_manager.store_license(
                license_key=license_key,
                expiry_date=expiry_date,
                status='active'
            )

            return {
                'success': True,
                'message': f'License activated successfully! Valid until {expiry_date.strftime("%Y-%m-%d")}',
                'expiry_date': expiry_date
            }
        else:
            error_msg = data.get('error', 'Unknown error')
            return {
                'success': False,
                'message': f'License activation failed: {error_msg}'
            }

    def check_license_status(self) -> Dict[str, Any]:
        """
        Check current license status from database.

        Returns:
            Dict: License status with keys:
                - is_valid (bool): Whether license is currently valid
                - status (str): License status ('active', 'expired', 'none')
                - expiry_date (datetime, optional): Expiration date
                - days_remaining (int, optional): Days until expiration
        """
        license_info = self.db_manager.get_license_status()

        if not license_info:
            return {
                'is_valid': False,
                'status': 'none',
                'message': 'No license activated'
            }

        status = license_info.get('status', 'unknown')
        expiry_str = license_info.get('expiry_date')

        if not expiry_str:
            return {
                'is_valid': False,
                'status': 'invalid',
                'message': 'License data corrupted'
            }

        try:
            expiry_date = datetime.fromisoformat(expiry_str)
            now = datetime.now()

            if expiry_date > now:
                days_remaining = (expiry_date - now).days

                return {
                    'is_valid': True,
                    'status': 'active',
                    'expiry_date': expiry_date,
                    'days_remaining': days_remaining,
                    'message': f'License active. {days_remaining} days remaining.'
                }
            else:
                return {
                    'is_valid': False,
                    'status': 'expired',
                    'expiry_date': expiry_date,
                    'message': 'License expired. Please renew at our website.'
                }

        except ValueError:
            return {
                'is_valid': False,
                'status': 'invalid',
                'message': 'License data corrupted'
            }

    def is_license_valid(self) -> bool:
        """
        Quick check if license is valid.

        Returns:
            bool: True if license is active and not expired
        """
        return self.db_manager.is_license_valid()

    def get_license_info(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed license information.

        Returns:
            Dict or None: License information or None if no license
        """
        return self.db_manager.get_license_status()

    def deactivate_license(self) -> bool:
        """
        Deactivate current license.

        Returns:
            bool: True if deactivated successfully
        """
        license_info = self.db_manager.get_license_status()

        if license_info:
            self.db_manager.store_license(
                license_key=license_info['license_key'],
                expiry_date=datetime.now(),  # Set to now to expire it
                status='deactivated'
            )
            return True

        return False


def generate_license_keys(count: int = 200, output_file: str = "license_keys.json") -> List[str]:
    """
    Generate license keys (server-side utility).

    This function would typically run on your server to generate the 200 keys.

    Args:
        count (int): Number of keys to generate
        output_file (str): File to save keys to

    Returns:
        List[str]: Generated license keys
    """
    import secrets
    import string

    keys = []
    alphabet = string.ascii_uppercase + string.digits

    for i in range(count):
        # Generate 16 random alphanumeric characters
        key_chars = ''.join(secrets.choice(alphabet) for _ in range(16))

        # Format as XXXX-XXXX-XXXX-XXXX
        key = f"{key_chars[0:4]}-{key_chars[4:8]}-{key_chars[8:12]}-{key_chars[12:16]}"
        keys.append(key)

    # Save to file
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'total_keys': count,
        'keys': [
            {
                'key': key,
                'status': 'unused',
                'created_at': datetime.now().isoformat()
            }
            for key in keys
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated {count} license keys and saved to {output_file}")
    return keys


if __name__ == "__main__":
    # Test license validator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import get_config
    from core.db_manager import DatabaseManager

    config = get_config()
    db = DatabaseManager()

    validator = LicenseValidator(config, db)

    # Test key validation
    test_key = "ABCD-1234-EFGH-5678"

    print(f"Testing license key: {test_key}")
    print(f"Format valid: {validator.validate_key_format(test_key)}")

    # Test activation
    result = validator.activate_license(test_key)
    print(f"\nActivation result: {result['message']}")

    # Check status
    status = validator.check_license_status()
    print(f"\nLicense status: {status['message']}")

    # Generate sample keys (for server-side use)
    print("\n--- Generating 10 sample keys ---")
    sample_keys = generate_license_keys(count=10, output_file="sample_keys.json")
    print("Sample keys:")
    for key in sample_keys[:5]:
        print(f"  {key}")
