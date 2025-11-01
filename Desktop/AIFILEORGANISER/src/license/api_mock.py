"""
License API Mock Server

Copyright (c) 2025 Alexandru Emanuel Vasile. All rights reserved.
Proprietary Software - 200-Key Limited Release License

This is a simple mock API server for license verification.
In production, this would be replaced with a proper backend service.

This mock server simulates the license verification endpoint and can be used
for testing without deploying a full backend.

NOTICE: This software is proprietary and confidential.
See LICENSE.txt for full terms and conditions.

Author: Alexandru Emanuel Vasile
License: Proprietary (200-key limited release)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
from pathlib import Path


# Request/Response models
class LicenseVerifyRequest(BaseModel):
    key: str


class LicenseVerifyResponse(BaseModel):
    valid: bool
    expiry: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None


# Mock license database (in production, use real database)
class MockLicenseDB:
    """Mock license database for testing."""

    def __init__(self, keys_file: str = "license_keys.json"):
        """
        Initialize mock database.

        Args:
            keys_file (str): Path to JSON file containing license keys
        """
        self.keys_file = Path(keys_file)
        self.licenses: Dict[str, Dict] = {}
        self._load_keys()

    def _load_keys(self):
        """Load license keys from file if it exists."""
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                data = json.load(f)
                for key_data in data.get('keys', []):
                    self.licenses[key_data['key']] = {
                        'status': key_data.get('status', 'unused'),
                        'created_at': key_data.get('created_at'),
                        'activated_at': None,
                        'expiry_date': None
                    }
        else:
            # Create some default test keys
            test_keys = [
                "ABCD-1234-EFGH-5678",
                "TEST-0000-1111-2222",
                "DEMO-AAAA-BBBB-CCCC"
            ]

            for key in test_keys:
                self.licenses[key] = {
                    'status': 'unused',
                    'created_at': datetime.now().isoformat(),
                    'activated_at': None,
                    'expiry_date': None
                }

    def _save_keys(self):
        """Save current license state to file."""
        data = {
            'updated_at': datetime.now().isoformat(),
            'total_keys': len(self.licenses),
            'keys': [
                {
                    'key': key,
                    **info
                }
                for key, info in self.licenses.items()
            ]
        }

        self.keys_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.keys_file, 'w') as f:
            json.dump(data, f, indent=2)

    def verify_key(self, license_key: str) -> Dict:
        """
        Verify a license key.

        Args:
            license_key (str): License key to verify

        Returns:
            Dict: Verification result
        """
        if license_key not in self.licenses:
            return {
                'valid': False,
                'message': 'License key not found'
            }

        license_info = self.licenses[license_key]
        status = license_info['status']

        # If unused, activate it
        if status == 'unused':
            expiry_date = datetime.now() + timedelta(days=30)

            license_info['status'] = 'active'
            license_info['activated_at'] = datetime.now().isoformat()
            license_info['expiry_date'] = expiry_date.isoformat()

            self._save_keys()

            return {
                'valid': True,
                'expiry': expiry_date.isoformat(),
                'status': 'active',
                'message': 'License activated successfully'
            }

        # If active, check expiry
        elif status == 'active':
            expiry_str = license_info.get('expiry_date')

            if expiry_str:
                expiry_date = datetime.fromisoformat(expiry_str)

                if expiry_date > datetime.now():
                    return {
                        'valid': True,
                        'expiry': expiry_str,
                        'status': 'active',
                        'message': 'License is active'
                    }
                else:
                    # Expired
                    license_info['status'] = 'expired'
                    self._save_keys()

                    return {
                        'valid': False,
                        'expiry': expiry_str,
                        'status': 'expired',
                        'message': 'License has expired'
                    }

        # Any other status (expired, revoked, etc.)
        return {
            'valid': False,
            'status': status,
            'message': f'License is {status}'
        }


# Create FastAPI app
app = FastAPI(
    title="AI File Organiser License API (Mock)",
    description="Mock license verification API for testing",
    version="1.0.0"
)

# Initialize mock database
license_db = MockLicenseDB()


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "AI File Organiser License API (Mock)",
        "version": "1.0.0",
        "endpoints": {
            "verify": "/api/verify-license"
        }
    }


@app.post("/api/verify-license", response_model=LicenseVerifyResponse)
def verify_license(request: LicenseVerifyRequest):
    """
    Verify a license key.

    Args:
        request: License verification request

    Returns:
        LicenseVerifyResponse: Verification result
    """
    result = license_db.verify_key(request.key)

    return LicenseVerifyResponse(**result)


@app.get("/api/stats")
def get_stats():
    """Get license statistics (admin endpoint)."""
    total = len(license_db.licenses)
    unused = sum(1 for info in license_db.licenses.values() if info['status'] == 'unused')
    active = sum(1 for info in license_db.licenses.values() if info['status'] == 'active')
    expired = sum(1 for info in license_db.licenses.values() if info['status'] == 'expired')

    return {
        'total_licenses': total,
        'unused': unused,
        'active': active,
        'expired': expired,
        'remaining': unused
    }


def run_mock_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the mock license server.

    Args:
        host (str): Host to bind to
        port (int): Port to listen on
    """
    import uvicorn

    print(f"""
    ===========================================
    AI File Organiser - Mock License API Server
    ===========================================

    Running on: http://{host}:{port}

    Endpoints:
    - POST /api/verify-license
    - GET  /api/stats

    Test keys available:
    - ABCD-1234-EFGH-5678
    - TEST-0000-1111-2222
    - DEMO-AAAA-BBBB-CCCC

    Press Ctrl+C to stop
    ===========================================
    """)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_mock_server()
