"""Resolve checked-in assets independently of the shell's current directory."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ARTIFACTS = REPO_ROOT / "artifacts"
WEB = PACKAGE_ROOT / "web"
