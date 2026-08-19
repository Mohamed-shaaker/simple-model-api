"""
tests/conftest.py - Shared pytest configuration and fixtures.
"""
import sys
import os

# Ensure the project root is on sys.path so imports like `from inference import`
# and `from main import` resolve correctly when pytest is run from any directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
