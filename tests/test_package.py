"""Packaging contract: the project metadata and the package namespace exist."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_the_runtime_dependencies():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = config["project"]
    assert project["name"] == "auon-optimizer-mlx"
    assert project["version"] == "0.1.0"
    assert project["requires-python"].startswith(">=3.10")
    dependencies = " ".join(project["dependencies"])
    assert "mlx" in dependencies
    assert "numpy" in dependencies
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_gitignore_covers_the_house_style_baseline():
    lines = set((ROOT / ".gitignore").read_text().splitlines())
    for entry in (
        "__pycache__/",
        "*.pyc",
        ".venv/",
        ".pytest_cache/",
        "artifacts/",
        "*.egg-info/",
    ):
        assert entry in lines


def test_readme_cites_the_paper_exactly():
    readme = (ROOT / "README.md").read_text()
    assert "AuON: A Linear-time Alternative to Orthogonal Momentum Updates" in readme
    assert re.search(r"arXiv:2509\.24320", readme)
    assert "not** a reproduction" in readme or "not a reproduction" in readme


def test_package_imports_and_reports_a_version():
    import auon

    assert auon.__version__ == "0.1.0"
