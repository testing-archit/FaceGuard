"""Smoke test: the whole experiment pipeline runs from one config and writes every output."""

import json
from pathlib import Path

import pytest
import yaml

from eval.run_experiments import BEGIN, END, main

CROSS = Path("data/samples/synthetic_cross.gt.json")
ER = Path("data/samples/synthetic_er.gt.json")


@pytest.mark.skipif(not (CROSS.exists() and ER.exists()), reason="run scripts/make_*_clip.py")
def test_one_command_regenerates_results(tmp_path):
    cfg = {
        "runs": 1,
        "videos": [{"truth": str(CROSS)}],
        "images": {"synthetic_stills": {"truth": str(ER), "every": 100},
                   "annotated": ["data/ground_truth.json"]},
        "e1_detectors": ["yunet"],
        "e2_intervals": [10],
        "e3_trackers": ["lk", "off"],
        "e6_paddings": [0.0, 0.25],
        "e7_unconfirmed_intervals": [3],
        "e8_paddings": [0.25], "e9_windows": [1, 3], "e10_trackers": ["lk"], "e11_tile_grids": [2],
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(cfg))
    md = tmp_path / "RESULTS.md"
    md.write_text(f"# R\n\n{BEGIN}\nSTALE-PLACEHOLDER\n{END}\n\n## kept by hand\n")
    assert main(["--config", str(tmp_path / "exp.yaml"), "--out", str(tmp_path), "--results-md", str(md)]) == 0

    text = md.read_text()
    assert "STALE-PLACEHOLDER" not in text and "## kept by hand" in text and "## E2" in text
    raw = json.loads((tmp_path / "raw_results.json").read_text())
    assert set(raw) >= {"e1", "e1_real", "e2", "e3", "e4", "e4_real", "e6", "e7", "e8", "e9", "e10", "e11", "environment"}
    for png in ("fps_vs_interval.png", "leakframes_vs_interval.png", "pr_curve.png",
                "condition_breakdown.png", "padding_sweep.png"):
        assert (tmp_path / png).stat().st_size > 0
