"""Honesty check: the README's experiment tables must quote the committed results.

Every numeric cell of the four tables in the README's "Experiment" section has
to be a rounding of a number that exists in ``results/experiment.json``, and
every headline metric in that file has to appear in the README.  Prose numbers
(ratios derived from the tables, and the paper's own benchmark numbers, which
are deliberately *not* reproduced here) are not covered by this test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results" / "experiment.json"

NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SEPARATOR = re.compile(r"^\|[\s:\-|]+\|$")


def readme_experiment_section() -> str:
    text = README.read_text()
    start = text.index("## Experiment")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def table_rows(section: str) -> list[list[str]]:
    """Cells of every data row of every markdown table in ``section``."""
    rows: list[list[str]] = []
    header_pending = False
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            header_pending = False
            continue
        if SEPARATOR.match(line):
            header_pending = False
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not rows and not header_pending:
            header_pending = True  # the first row of a table is its header
            continue
        rows.append(cells)
    return rows


def all_numbers(node, out: set[float]) -> set[float]:
    if isinstance(node, bool):
        return out
    if isinstance(node, (int, float)):
        out.add(float(node))
    elif isinstance(node, dict):
        for value in node.values():
            all_numbers(value, out)
    elif isinstance(node, list):
        for value in node:
            all_numbers(value, out)
    return out


def tolerance(token: str) -> float:
    """Half a unit in the last digit the README actually wrote."""
    mantissa, _, exponent = token.lower().partition("e")
    exponent = int(exponent) if exponent else 0
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 0.5 * 10.0 ** (exponent - decimals)


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS.read_text())


def test_every_readme_table_number_comes_from_the_results(results):
    available = all_numbers(results, set())
    unmatched: list[tuple[str, float]] = []
    for cells in table_rows(readme_experiment_section()):
        for cell in cells:
            if cell.startswith("`") or not NUMBER.search(cell):
                continue
            stripped = cell.replace("*", "").replace("\\|", "").replace("|", "")
            # only cells that are purely a number (or a number +- number) count
            if not re.fullmatch(r"[\s\d.,eE+\-]+", stripped):
                continue
            for token in NUMBER.findall(stripped):
                value = float(token)
                tol = tolerance(token)
                if not any(abs(value - candidate) <= tol for candidate in available):
                    unmatched.append((cell, value))
    assert not unmatched, f"README table numbers not found in results/: {unmatched[:8]}"


def test_every_headline_result_is_quoted_in_the_readme(results):
    readme = README.read_text()
    missing: list[str] = []
    for arm, entry in results["table"].items():
        for key in ("final_loss", "final_accuracy", "peak_attn_logit", "ms_per_step"):
            value = entry[key]["mean"]
            if key in ("final_loss", "final_accuracy"):
                rendered = f"{value:.4f}"
            elif key == "ms_per_step":
                rendered = f"{value:.2f}"
            else:
                rendered = f"{value:.1f}"
            if rendered not in readme:
                missing.append(f"{arm}.{key}={rendered}")
    for arm, entry in results["scaling"]["arms"].items():
        for ms in entry["ms_per_step"]:
            if f"{ms:.3f}" not in readme:
                missing.append(f"scaling.{arm}={ms:.3f}")
    for arm, entry in results["brake"].items():
        if f"{entry['peak_attn_logit']:.2f}" not in readme:
            missing.append(f"brake.{arm}.peak={entry['peak_attn_logit']:.2f}")
    assert not missing, f"results not quoted in the README: {missing[:8]}"


def test_the_papers_benchmarks_appear_only_in_the_not_reproduced_bullet():
    """The paper's own numbers are named as the paper's, and never as ours."""
    readme = README.read_text()
    start = readme.index("**Nothing here reproduces the paper's benchmarks.**")
    end = readme.index("\n* ", start)
    block = readme[start:end]
    for quoted in ("4.2882", "4.4349", "74.42", "65.99"):
        assert quoted in block, quoted
        assert readme.count(quoted) == block.count(quoted), quoted
    assert "not** a reproduction" in readme or "not a reproduction" in readme
