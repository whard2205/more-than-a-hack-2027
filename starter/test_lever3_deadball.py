"""Smallest check that would fail if the dead-ball detector broke. Synthetic, no match data.

    python starter/test_lever3_deadball.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from baseline import HOME, NOBODY  # noqa: E402
from lever1_tagging import DEAD  # noqa: E402
from lever3_deadball import apply_deadball, detect_dead  # noqa: E402


def _signals(n=1000, fast=4.0):
    ball_x = np.full(n, 0.5, dtype=np.float32)
    ball_y = np.full(n, 0.5, dtype=np.float32)
    speed = np.full(n, fast, dtype=np.float32)
    return ball_x, ball_y, speed


def test_ball_leaving_the_pitch_is_dead_with_padding():
    bx, by, sp = _signals()
    bx[500:510] = 1.2  # ten frames outside the pitch
    mask = detect_dead(bx, by, sp, ball_pad=20)
    assert mask[500:510].all()
    assert mask[480] and mask[529]          # padded on both sides
    assert not mask[400] and not mask[600]  # and nowhere else


def test_short_slow_spell_is_not_dead_but_a_long_one_is():
    bx, by, sp = _signals()
    sp[300:350] = 0.5   # 50 frames slow: a quiet build-up
    sp[600:900] = 0.5   # 300 frames slow: a real stoppage
    mask = detect_dead(bx, by, sp, min_slow=150)
    assert not mask[300:350].any(), "a 2-second lull was marked dead"
    assert mask[650:850].all(), "a 12-second stop was missed"


def test_gradual_slowdown_is_not_a_stoppage():
    # The mistake a fixed threshold makes: players tire, speed drifts from 4.0 to 1.6 m/s over the
    # match with no stoppage anywhere. Nothing may be flagged.
    n = 30000
    bx, by, _ = _signals(n)
    sp = np.linspace(4.0, 1.6, n).astype(np.float32)
    assert not detect_dead(bx, by, sp).any()


def test_unobserved_speed_and_ball_do_not_crash_or_flag():
    bx, by, sp = _signals()
    bx[:] = np.nan
    by[:] = np.nan
    sp[100:200] = np.nan
    mask = detect_dead(bx, by, sp)
    assert not mask.any()


def test_apply_overwrites_only_masked_frames_and_keeps_existing_dead():
    codes = np.array([HOME, HOME, NOBODY, DEAD, HOME], dtype=np.int8)
    mask = np.array([False, True, True, False, False])
    out = apply_deadball(codes, mask)
    assert list(out) == [HOME, DEAD, DEAD, DEAD, HOME]
    assert list(codes) == [HOME, HOME, NOBODY, DEAD, HOME], "input was modified"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  " + test.__name__)
    print("\n{} checks passed".format(len(tests)))
