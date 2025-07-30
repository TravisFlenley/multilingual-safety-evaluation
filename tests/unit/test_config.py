"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils import ConfigManager, Config, get_config, get_config_manager


class TestConfig:
    """Test Config class functionality."""
    
    def test_config_creation(self):
        """Test creating config from dictionary."""
        data = {
            "api_keys": {
                "openai": "test-key",
                "anthropic": "test-key-2"
            },
            "models": {
                "openai": {
                    "models": ["gpt-3.5-turbo"],
                    "temperature": 0.7
                }
            }
        }
        
        config = Config(data)
        
        assert config.api_keys.openai == "test-key"
        assert config.api_keys.anthropic == "test-key-2"
        assert config.models.openai.models == ["gpt-3.5-turbo"]
        assert config.models.openai.temperature == 0.7
        
    def test_get_with_dot_notation(self):
        """Test getting values with dot notation."""
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        config = Config(data)
        
        assert config.get("level1.level2.level3") == "value"
        assert config.get("level1.level2") == {"level3": "value"}
        assert config.get("nonexistent", "default") == "default"
        
    def test_set_with_dot_notation(self):
        """Test setting values with dot notation."""
        config = Config({"existing": "value"})
        
        config.set("new.nested.value", 42)
        assert config.new.nested.value == 42
        assert config.get("new.nested.value") == 42
        
        config.set("existing", "updated")
        assert config.existing == "updated"
        
    def test_dictionary_access(self):
        """Test dictionary-style access."""
        config = Config({"key": "value"})
        
        assert config["key"] == "value"
        
        config["new_key"] = "new_value"
        assert config["new_key"] == "new_value"
        assert config.new_key == "new_value"
        
    def test_to_dict(self):
        """Test conversion back to dictionary."""
        data = {"a": 1, "b": {"c": 2}}
        config = Config(data)
        
        result = config.to_dict()
        assert result == data


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "api_keys": {
                    "openai": "test-openai-key",
                    "anthropic": "test-anthropic-key"
                },
                "models": {
                    "openai": {
                        "models": ["gpt-3.5-turbo", "gpt-4"],
                        "default_params": {
                            "temperature": 0.8,
                            "max_tokens": 500
                        }
                    }
                },
                "safety_thresholds": {
                    "harmful_content": 0.85,
                    "bias": 0.75
                },
                "languages": {
                    "supported": ["en", "es", "zh"],
                    "default": "en"
                }
            }
            yaml.dump(config_data, f)
            return Path(f.name)
            
    def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        manager = ConfigManager(temp_config_file)
        
        assert manager.config.api_keys.openai == "test-openai-key"
        assert "gpt-4" in manager.config.models.openai.models
        assert manager.config.safety_thresholds.harmful_content == 0.85
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_load_missing_config(self):
        """Test loading with missing config file."""
        manager = ConfigManager("nonexistent.yaml")
        
        # Should load defaults
        assert hasattr(manager.config, "api_keys")
        assert hasattr(manager.config, "models")
        assert hasattr(manager.config, "evaluation")
        
    def test_merge_with_defaults(self, temp_config_file):
        """Test merging user config with defaults."""
        manager = ConfigManager(temp_config_file)
        
        # User-specified values
        assert manager.config.api_keys.openai == "test-openai-key"
        
        # Default values not in user config
        assert hasattr(manager.config, "evaluation")
        assert manager.config.evaluation.batch_size == 32
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_get_api_key(self, temp_config_file):
        """Test getting API keys."""
        manager = ConfigManager(temp_config_file)
        
        assert manager.get_api_key("openai") == "test-openai-key"
        assert manager.get_api_key("anthropic") == "test-anthropic-key"
        assert manager.get_api_key("nonexistent") is None
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_get_model_config(self, temp_config_file):
        """Test getting model configuration."""
        manager = ConfigManager(temp_config_file)
        
        openai_config = manager.get_model_config("openai")
        assert "models" in openai_config
        assert "gpt-3.5-turbo" in openai_config["models"]
        assert openai_config["default_params"]["temperature"] == 0.8
        
        # Non-existent provider
        empty_config = manager.get_model_config("nonexistent")
        assert empty_config == {}
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_get_supported_languages(self, temp_config_file):
        """Test getting supported languages."""
        manager = ConfigManager(temp_config_file)
        
        languages = manager.get_supported_languages()
        assert languages == ["en", "es", "zh"]
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_get_supported_languages_dict_format(self):
        """Test getting languages when stored as dict."""
        manager = ConfigManager()
        manager.config.languages.supported = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"}
        ]
        
        languages = manager.get_supported_languages()
        assert languages == ["en", "es"]
        
    def test_get_safety_threshold(self, temp_config_file):
        """Test getting safety thresholds."""
        manager = ConfigManager(temp_config_file)
        
        # Direct threshold
        assert manager.get_safety_threshold("harmful_content") == 0.85
        assert manager.get_safety_threshold("bias") == 0.75
        
        # Nested threshold
        manager.config.safety_thresholds.harmful_content = {
            "violence": 0.9,
            "self_harm": 0.95
        }
        
        assert manager.get_safety_threshold("harmful_content", "violence") == 0.9
        assert manager.get_safety_threshold("harmful_content", "self_harm") == 0.95
        
        # Default value
        assert manager.get_safety_threshold("nonexistent") == 0.7
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_update_config(self, temp_config_file):
        """Test updating configuration."""
        manager = ConfigManager(temp_config_file)
        
        # Update existing value
        manager.update_config({"evaluation.batch_size": 64})
        assert manager.config.evaluation.batch_size == 64
        
        # Add new value
        manager.update_config({"new_setting.enabled": True})
        assert manager.config.new_setting.enabled is True
        
        # Cleanup
        temp_config_file.unlink()
        
    def test_save_config(self, temp_config_file):
        """Test saving configuration."""
        manager = ConfigManager(temp_config_file)
        
        # Modify config
        manager.update_config({"test_value": 42})
        
        # Save to new file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            save_path = Path(f.name)
            
        manager.save_config(save_path)
        
        # Load saved config
        with open(save_path) as f:
            saved_data = yaml.safe_load(f)
            
        assert saved_data["test_value"] == 42
        assert saved_data["api_keys"]["openai"] == "test-openai-key"
        
        # Cleanup
        temp_config_file.unlink()
        save_path.unlink()
        
    def test_config_validation(self, temp_config_file):
        """Test configuration validation."""
        manager = ConfigManager(temp_config_file)
        
        # Add invalid threshold
        manager.config.safety_thresholds.invalid = 1.5
        
        # Validation should handle this gracefully
        manager._validate_config()  # Should log warning but not crash
        
        # Cleanup
        temp_config_file.unlink()


class TestGlobalConfig:
    """Test global configuration functions."""
    
    def test_get_config(self):
        """Test getting global config instance."""
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        assert config1 is config2
        
    def test_get_config_manager(self):
        """Test getting global config manager."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        # Should return same instance
        assert manager1 is manager2