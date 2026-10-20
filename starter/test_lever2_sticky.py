"""Smallest check that would fail if the hysteresis state machine broke.

No framework, no fixtures -- assert-based, synthetic sequences plus one check against the real
scoreboard once it exists.

    python starter/test_lever2_sticky.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline import HOME, AWAY, NOBODY  # noqa: E402
from lever1_tagging import DEAD  # noqa: E402
from lever2_sticky import sticky_labels  # noqa: E402


def test_short_flicker_does_not_flip():
    # 20 frames of HOME, then a 3-frame AWAY blip (shorter than any threshold we ship), then HOME
    # again. Possession must never read "A" anywhere -- that blip is exactly what CHALLENGE.md
    # says doesn't count (a touch without control).
    codes = np.array([HOME] * 20 + [AWAY] * 3 + [HOME] * 20, dtype=np.int8)
    labels = sticky_labels(codes, threshold=25)
    assert "A" not in labels, "a 3-frame blip flipped possession at threshold 25"
    assert set(labels) == {"H"}


def test_sustained_evidence_flips_exactly_at_the_threshold():
    threshold = 10
    codes = np.array([HOME] * 20 + [AWAY] * 20, dtype=np.int8)
    labels = sticky_labels(codes, threshold)
    # The flip must land inside the AWAY run, once `threshold` AWAY frames have been seen --
    # not before (that would mean it's not debouncing) and not much later (that would mean it's
    # not using `threshold` at all).
    first_a = np.where(labels == "A")[0][0]
    assert 20 <= first_a < 20 + threshold + 1, \
        "flip at {} is not where a {}-frame threshold should land".format(first_a, threshold)


def test_dead_always_outputs_d_and_resets_state():
    codes = np.array([HOME] * 30 + [DEAD] * 5 + [NOBODY] * 10 + [AWAY] * 30, dtype=np.int8)
    labels = sticky_labels(codes, threshold=10)
    assert np.all(labels[30:35] == "D"), "DEAD frames must always be D"
    # held as D through the NOBODY gap too, since state was reset and nobody has restarted yet
    assert np.all(labels[35:45] == "D"), "no evidence yet after a stoppage should still read D"
    assert labels[45] == "A", "first confident read after a stoppage should win immediately"


def test_output_shape_and_alphabet():
    codes = np.array([HOME, AWAY, NOBODY, DEAD] * 50, dtype=np.int8)
    labels = sticky_labels(codes, threshold=5)
    assert len(labels) == len(codes)
    assert set(labels) <= {"H", "A", "D"}


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  " + test.__name__)
    print("\n{} checks passed".format(len(tests)))
