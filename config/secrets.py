# config/secrets.py
import os
from pathlib import Path
from typing import Dict, Any
import json

class SecretsManager:
    """Manages application secrets and configuration"""
    
    _instance = None
    _secrets: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecretsManager, cls).__new__(cls)
            cls._instance._load_secrets()
        return cls._instance
    
    def _load_secrets(self) -> None:
        """Load secrets from secrets.json file"""
        secrets_path = Path(__file__).parent / 'secrets.json'
        
        if not secrets_path.exists():
            self._create_default_secrets(secrets_path)
        
        try:
            with open(secrets_path) as f:
                self._secrets = json.load(f)
        except Exception as e:
            print(f"Error loading secrets: {e}")
            self._secrets = {}
    
    def _create_default_secrets(self, path: Path) -> None:
        """Create default secrets.json file"""
        default_secrets = {
            "ALPHA_VANTAGE_API_KEY": "your_api_key_here",
            "API_BASE_URL": "https://www.alphavantage.co/query"
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(default_secrets, f, indent=2)
            print(f"Created default secrets file at {path}")
        except Exception as e:
            print(f"Error creating default secrets file: {e}")
    
    @property
    def alpha_vantage_key(self) -> str:
        """Get Alpha Vantage API key"""
        return self._secrets.get("ALPHA_VANTAGE_API_KEY", "")
    
    @property
    def api_base_url(self) -> str:
        """Get API base URL"""
        return self._secrets.get("API_BASE_URL", "https://www.alphavantage.co/query")

# config/__init__.py
from .secrets import SecretsManager

secrets = SecretsManager()