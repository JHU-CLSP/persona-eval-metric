"""CLI package for persona-eval.

The console-script entry point ``persona-eval = "persona_eval.cli:main"``
resolves through this re-export.
"""

from persona_eval.cli.main import main

__all__ = ["main"]
