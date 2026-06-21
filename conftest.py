"""Ensures the project root is importable so `import cgsp` works under pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
