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


def test_sustained_evidence_flip_is_back_dated_to_the_streak_start():
    threshold = 10
    codes = np.array([HOME] * 20 + [AWAY] * 20, dtype=np.int8)
    labels = sticky_labels(codes, threshold)
    # The AWAY possession truly started at frame 20. Without back-dating the label would only turn
    # at 29 (when the threshold clears) -- a lag the scorer charges on every transition.
    assert np.all(labels[:20] == "H")
    assert np.all(labels[20:] == "A"), \
        "flip not back-dated: first A at {}".format(np.where(labels == "A")[0][0])


def test_back_dating_covers_an_invisible_ball_inside_the_streak():
    # AWAY evidence, a stretch with the ball invisible, then more AWAY until the threshold clears.
    # The whole stretch from the first AWAY frame belongs to AWAY.
    codes = np.array([HOME] * 20 + [AWAY] * 4 + [NOBODY] * 6 + [AWAY] * 10, dtype=np.int8)
    labels = sticky_labels(codes, threshold=10)
    assert np.all(labels[:20] == "H")
    assert np.all(labels[20:] == "A")


def test_streak_broken_by_the_holder_does_not_back_date():
    # AWAY builds 8 frames (short of 10), HOME touches once, AWAY builds 8 again: never flips,
    # so nothing may be rewritten either.
    codes = np.array([HOME] * 20 + [AWAY] * 8 + [HOME] + [AWAY] * 8 + [HOME] * 5, dtype=np.int8)
    labels = sticky_labels(codes, threshold=10)
    assert set(labels) == {"H"}


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
