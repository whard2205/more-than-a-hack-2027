"""Lever 1: use Smart Tagging to confirm dead-ball frames before smoothing.

Same nearest-player signal as baseline.py. The only addition: a NOBODY run (ball not visible --
43% of graded frames) that falls inside a tagged SET PIECES window IS the stoppage itself, so mark
it DEAD, a sentinel that the majority-vote smoothing skips over and that is always written as 'D'
regardless of what its neighbours voted. Baseline has no such signal -- it only knows "ball not
visible", which inside a wide smoothing window (+/-125 frames) can get voted into the wrong team's
label right across a real stoppage.

Why not also force the restart team onto the following frames: measured on this match, every
NOBODY run inside a tagged window ends exactly where the ball becomes visible again -- by
definition of "run", there is no still-ambiguous frame left afterwards to hand to anyone. Tracking
already resumes with a confident read right at the restart, so there is nothing here to correct.
(Team tags would also only cover 27 of 59 instances anyway -- all 26 throw-ins and both
match-opening kickoffs carry Team="N/A" in this file.)

Why not paint the whole tagged interval: every non-kickoff SET PIECES interval is exactly 875
frames (35 s) -- a fixed highlight-clip length, not the real dead-ball span (only the two Kick Off
tags are tight, ~125 frames / 5 s). A corner's clip runs long after the corner is taken, so forcing
the whole window to one team would overwrite real, possibly contested, play with a guess. Only
touching NOBODY frames means this can only improve frames the baseline was already blind on --
never override a confident tracked read.

    python starter/lever1_tagging.py                  # writes 4 submissions
    python evaluation/run_evaluation.py                # then score them

ponytail: this only uses the 638 Smart Tagging instances to locate stoppages, never their Team tag
-- restart ownership (lever 2) needs a different signal (e.g. a sticky state machine), not more
Smart Tagging parsing. Revisit if spellF1 still loses possessions at set-piece boundaries.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, require  # noqa: E402
from baseline import NOBODY, LABELS, VARIANTS, build, smooth, write_submission, RESULTS_DIR  # noqa: E402
import load_smart_tagging  # noqa: E402

DEAD = 2  # sentinel: definitely a stoppage, never counted as a vote and never smoothed away
SET_PIECES_CODE = "set pieces"


def apply_smart_tagging(codes, game=GAME, data_dir=None):
    """codes, with DEAD written over every NOBODY run that falls inside a tagged set piece."""
    events = load_smart_tagging.load(game, data_dir)
    set_pieces = [e for e in events if (e["code"] or "").strip().lower() == SET_PIECES_CODE]

    out = codes.copy()
    n_frames = len(codes)
    dead_marked = 0

    for sp in set_pieces:
        start = max(0, sp["start_frame"])
        end = min(n_frames, sp["end_frame"])
        if start >= end:
            continue
        window = out[start:end]
        is_nobody = (window == NOBODY)
        window[is_nobody] = DEAD
        dead_marked += int(np.sum(is_nobody))

    return out, {"dead_marked": dead_marked, "n_set_pieces": len(set_pieces)}


def smooth_with_dead(codes, window):
    """Like baseline.smooth(), but DEAD frames are forced to 'D' after the vote, not during it."""
    is_dead = (codes == DEAD)
    vote_codes = np.where(is_dead, NOBODY, codes)  # DEAD doesn't count as a vote either way
    labels = smooth(vote_codes, window)
    labels[is_dead] = "D"
    return labels


if __name__ == '__main__':
    codes = build()
    codes, stats = apply_smart_tagging(codes)
    print("smart tagging: {dead_marked} frames confirmed dead across {n_set_pieces} "
          "tagged set pieces".format(**stats))
    print()

    for name, window in VARIANTS:
        labels = smooth_with_dead(codes, window)
        path = write_submission(
            os.path.join(RESULTS_DIR, "tagging_{}.csv".format(name.replace("baseline_", ""))),
            labels)
        counts = {label: int(np.sum(labels == label)) for label in LABELS}
        print("{:<24} window +/-{:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name.replace("baseline", "tagging"), window, counts["H"], counts["A"], counts["D"]))
        print("{:<24} -> {}".format("", os.path.relpath(path, ROOT)))

    print("\nNow score them:  python evaluation/run_evaluation.py")
