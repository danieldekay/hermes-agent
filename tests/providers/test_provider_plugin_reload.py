"""Test for provider plugin reload behavior - reproduces the stale module mixing issue.

This test demonstrates the bug where updated model-provider plugins are never
reloaded in a running process, and lazy imports mix old and new plugin files.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest


PLUGIN_V1 = textwrap.dedent(
    """
    # v1: Profile class with lazy import in create_client
    from providers import register_provider
    from providers.base import ProviderProfile

    class TestReloadProfile(ProviderProfile):
        def create_client(self, **client_kwargs):
            # Lazy import - this will load the NEW client.py after update
            from .client import TestClient
            from .setup import get_config
            config = get_config()
            return TestClient(config)

    _PROFILE = TestReloadProfile(
        name="test-reload",
        aliases=("test-reload-alias",),
        env_vars=("TEST_RELOAD_KEY",),
        base_url="https://v1.example.com/v1",
        auth_type="api_key",
    )
    register_provider(_PROFILE)
    """
)

SETUP_V1 = textwrap.dedent(
    """
    PLUGIN_VERSION = "1.0"
    
    def get_config():
        return {"version": PLUGIN_VERSION, "endpoint": "v1"}
    """
)

CLIENT_V1 = textwrap.dedent(
    """
    class TestClient:
        def __init__(self, config):
            self.config = config
            self.version = config["version"]
    """
)


PLUGIN_V2 = textwrap.dedent(
    """
    # v2: Profile class with lazy import in create_client (NEW version)
    from providers import register_provider
    from providers.base import ProviderProfile

    class TestReloadProfile(ProviderProfile):
        def create_client(self, **client_kwargs):
            # Lazy import - this loads the NEW client.py
            from .client import TestClient
            from .setup import get_config
            config = get_config()
            return TestClient(config)

    _PROFILE = TestReloadProfile(
        name="test-reload",
        aliases=("test-reload-alias",),
        env_vars=("TEST_RELOAD_KEY",),
        base_url="https://v2.example.com/v1",
        auth_type="api_key",
    )
    register_provider(_PROFILE)
    """
)

SETUP_V2 = textwrap.dedent(
    """
    PLUGIN_VERSION = "2.0"
    # v2 adds a new function that client.py needs
    def get_config():
        return {"version": PLUGIN_VERSION, "endpoint": "v2", "new_param": "required"}
    
    def drop_api_credentials():  # New in v2
        return "dropped"
    """
)

CLIENT_V2 = textwrap.dedent(
    """
    # v2 client needs drop_api_credentials from setup
    from .setup import drop_api_credentials
    
    class TestClient:
        def __init__(self, config):
            self.config = config
            self.version = config["version"]
            self.dropped = drop_api_credentials()  # This will fail if old setup is loaded
    """
)


def _write_plugin(plugin_dir: Path, init_content: str, setup_content: str, client_content: str) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: test-reload\nkind: model-provider\nversion: 1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    (plugin_dir / "setup.py").write_text(setup_content, encoding="utf-8")
    (plugin_dir / "client.py").write_text(client_content, encoding="utf-8")


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    _pkg._HOME_LAYERS.clear()
    # Evict any cached plugin modules so the next import re-executes.
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


def test_provider_plugin_reload_picks_up_new_code(tmp_path, monkeypatch):
    """Test that updating a provider plugin and forcing reload picks up new code.
    
    This is the bug: after updating a plugin on disk, the next discovery
    should reload the plugin module and pick up new code. Currently it
    returns early because the module is already in sys.modules.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    import providers
    
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    
    plugin_dir = hermes_home / "plugins" / "model-providers" / "test-reload"
    
    # Install v1
    _write_plugin(plugin_dir, PLUGIN_V1, SETUP_V1, CLIENT_V1)
    _clear_provider_caches()
    
    # Load v1
    token = set_hermes_home_override(hermes_home)
    try:
        profile = providers.get_provider_profile("test-reload")
        assert profile is not None, "v1 plugin should be discovered"
        assert profile.base_url == "https://v1.example.com/v1"
        
        # Call create_client to trigger lazy import of client.py
        client = profile.create_client()
        assert client.version == "1.0", f"Expected v1 client, got {client.version}"
    finally:
        reset_hermes_home_override(token)
    
    # Update to v2 on disk
    _write_plugin(plugin_dir, PLUGIN_V2, SETUP_V2, CLIENT_V2)
    
    # Force reload by clearing caches and re-discovering
    _clear_provider_caches()
    
    # Load v2
    token = set_hermes_home_override(hermes_home)
    try:
        profile = providers.get_provider_profile("test-reload")
        assert profile is not None, "v2 plugin should be discovered after reload"
        assert profile.base_url == "https://v2.example.com/v1", f"Expected v2 base_url, got {profile.base_url}"
        
        # This should work with the new code, but currently fails because:
        # - Old __init__ and setup are still in sys.modules
        # - New client.py is loaded (lazy import) but imports from OLD setup
        # - Old setup doesn't have drop_api_credentials
        client = profile.create_client()
        assert client.version == "2.0", f"Expected v2 client, got {client.version}"
        assert client.dropped == "dropped", "v2 client should call new setup function"
    finally:
        reset_hermes_home_override(token)


def test_provider_plugin_stamp_tracks_file_changes(tmp_path, monkeypatch):
    """Test that _plugin_dir_stamps detects file changes inside plugin directory.
    
    Currently _plugin_dir_stamps only watches directory mtimes, not file changes.
    This means updating a file inside an existing plugin directory doesn't
    trigger a reload.
    """
    import providers
    
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    
    plugin_dir = hermes_home / "plugins" / "model-providers" / "test-stamp"
    _write_plugin(plugin_dir, PLUGIN_V1, SETUP_V1, CLIENT_V1)
    _clear_provider_caches()
    
    # Initial stamp
    stamp1 = providers._plugin_dir_stamps(hermes_home)
    
    # Modify a file inside the plugin dir (not the directory itself)
    import time
    time.sleep(0.01)  # Ensure mtime changes
    (plugin_dir / "client.py").write_text(CLIENT_V2, encoding="utf-8")
    
    stamp2 = providers._plugin_dir_stamps(hermes_home)
    
    # Currently this fails because stamps only track directory mtimes
    # After fix, stamp should change when files inside change
    assert stamp1 != stamp2, "Plugin directory stamps should detect file changes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])