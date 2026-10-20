"""Smallest check that would fail if lever1_tagging.py's logic broke.

No framework, no fixtures -- assert-based, run against the real data already on disk.

    python starter/test_lever1_tagging.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline import NOBODY, build  # noqa: E402
from lever1_tagging import DEAD, apply_smart_tagging, smooth_with_dead  # noqa: E402
import load_smart_tagging  # noqa: E402


def test_dead_only_marks_frames_that_were_nobody_inside_a_tagged_window():
    codes = build()
    tagged, stats = apply_smart_tagging(codes)

    changed = np.where(codes != tagged)[0]
    assert len(changed) == stats["dead_marked"] > 0, "nothing was marked dead -- parsing broke"
    assert np.all(codes[changed] == NOBODY), "a confident tracked frame got overwritten"
    assert np.all(tagged[changed] == DEAD), "a changed frame ended up as something other than DEAD"

    # Every changed frame must fall inside at least one tagged SET PIECES window.
    events = load_smart_tagging.load()
    windows = [(e["start_frame"], e["end_frame"]) for e in events
               if (e["code"] or "").strip().lower() == "set pieces"]
    for frame in changed[:: max(1, len(changed) // 200)]:  # sample, not all -- this is a smoke test
        assert any(s <= frame < en for s, en in windows), \
            "frame {} marked dead but is outside every tagged window".format(frame)


def test_smooth_with_dead_always_outputs_d_for_dead_frames():
    codes = build()
    tagged, _ = apply_smart_tagging(codes)
    for window in (0, 25, 50, 125):
        labels = smooth_with_dead(tagged, window)
        assert np.all(labels[tagged == DEAD] == "D")
        assert set(labels) <= {"H", "A", "D"}  # no stray label leaks through


def test_never_regresses_macro_f1_on_the_public_window():
    """The one claim this lever makes: smoothed variants should not score worse than baseline."""
    import subprocess
    root = os.path.dirname(HERE)
    # Only meaningful if both sets of CSVs already exist (starter/baseline.py and
    # starter/lever1_tagging.py both run) -- this is a smoke test, not a build step.
    import csv
    scoreboard = os.path.join(root, "evaluation", "scoreboard.csv")
    if not os.path.exists(scoreboard):
        print("  (skipped: run both baselines + evaluation first)")
        return
    rows = {r["submission"]: r for r in csv.DictReader(open(scoreboard))}
    for suffix in ("smooth25", "smooth50", "smooth125"):
        base, lever = rows.get("baseline_" + suffix), rows.get("tagging_" + suffix)
        if base is None or lever is None:
            continue
        assert float(lever["macro_f1"]) >= float(base["macro_f1"]) - 1e-9, \
            "tagging_{} regressed macroF1 vs baseline".format(suffix)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  " + test.__name__)
    print("\n{} checks passed".format(len(tests)))
