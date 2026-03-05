"""
tests/conftest.py — Shared pytest fixtures.
"""

import os
import pytest

# Set a dummy API key so Settings validation passes during tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")
