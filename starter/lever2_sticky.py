"""Lever 2: a sticky possession state machine, instead of a symmetric majority vote.

Built on lever1_tagging's codes (nearest-player evidence, with real stoppages already marked
DEAD). The remaining error, visible in evaluation/reports/*_transitions.csv for the smoothed
baselines: in a scramble, nearest-player flips between teams several times a second even though
nobody gained control -- CHALLENGE.md says exactly this doesn't count as a change of possession
("a single touch without control ... a won aerial duel where the ball is not controlled"). A
symmetric window smooths that noise but in doing so also blurs every real transition by the same
amount and merges short possessions together (smooth125's possessions run 27% longer than the
true 14.2 s average).

A hysteresis filter instead: possession only flips once the OTHER team has had `n` consecutive
frames of nearest-player evidence -- not a vote over a wide window, just "how long has the
challenger been true". A one- or two-frame flicker can never flip it; a real sustained
turnover (a clean tackle, an intercepted pass) does, as soon as the evidence holds up, not half a
window later. NOBODY frames (ball not visible) and DEAD frames neither reinforce nor challenge --
they just hold, which is the documented-safe default (CHALLENGE.md: a deflected pass that isn't
intercepted isn't a change either, and we have no evidence at all while the ball is invisible). A
DEAD frame also resets the challenger count and, this is the one special case, sets state to
"nobody yet" so the next possession is only assigned once real evidence appears after the restart
-- not inherited from whoever had it before the stoppage.

    python starter/lever2_sticky.py                    # writes a submission per threshold
    python evaluation/run_evaluation.py --bootstrap 1000

Measured on the public 5-minute window: spellF1 rises smoothly from threshold 5 to a flat plateau
at 50-65 frames (2.0-2.6 s) -- poss=13, duration=22.2s, fragmentation=0.71, spellF1 0.45-0.47 at
every point in that plateau -- then falls off past 75. A flat multi-point plateau rather than one
lucky value is the evidence that this is a real effect of the mechanism, not a fit to 17 possessions
(the same check EVALUATION.md asks for). It beats tagging_smooth125 (spellF1 0.4485) on this
window; still below the public window's own noise band, so treat it as the better-evidenced
candidate, not a confirmed winner -- the private full-match score decides that.

ponytail: thresholds are a flat frame count per class, same hysteresis for home and away and for
every match situation; a felt foul (whistle before the tackle lands) needs a shorter threshold than
a contested header, if the per-class confusion ever points that way specifically.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from baseline import HOME, AWAY, NOBODY, LABELS, build, write_submission, RESULTS_DIR  # noqa: E402
from lever1_tagging import DEAD, apply_smart_tagging  # noqa: E402

# Consecutive frames of evidence for the other side needed before possession actually flips.
# 25 fps. Brackets the plateau found by sweeping 5-200 (see module docstring) rather than a single
# hand-picked value -- score all three, don't commit to one off 17 public possessions
# (EVALUATION.md's warning, same one baseline.py's docstring repeats).
THRESHOLDS = (25, 50, 100)


def sticky_labels(codes, threshold):
    """One pass, O(n): a possession only changes after `threshold` straight frames back it."""
    out = np.empty(len(codes), dtype="<U1")
    state = None          # None = no current possessor (before first possession / after a DEAD)
    challenger = None      # the side currently building a streak against `state`
    streak = 0

    for i, code in enumerate(codes):
        if code == DEAD:
            state, challenger, streak = None, None, 0
            out[i] = "D"
            continue
        if code == NOBODY:
            out[i] = "D" if state is None else ("H" if state == HOME else "A")
            continue

        if code == state:
            challenger, streak = None, 0
        elif code == challenger:
            streak += 1
            if streak >= threshold:
                state, challenger, streak = code, None, 0
        else:
            challenger, streak = code, 1
            if state is None and streak >= 1:
                # Nobody to challenge yet (match just started, or just came out of a stoppage) --
                # the first confident read after "no possessor" wins immediately, no debounce.
                state, challenger, streak = code, None, 0

        out[i] = "D" if state is None else ("H" if state == HOME else "A")
    return out


if __name__ == '__main__':
    codes = build()
    codes, stats = apply_smart_tagging(codes)
    print("smart tagging: {dead_marked} frames confirmed dead".format(**stats))
    print()

    for threshold in THRESHOLDS:
        labels = sticky_labels(codes, threshold)
        name = "sticky_h{}".format(threshold)
        path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        counts = {label: int(np.sum(labels == label)) for label in LABELS}
        print("{:<16} threshold {:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name, threshold, counts["H"], counts["A"], counts["D"]))
        print("{:<16} -> {}".format("", os.path.relpath(path, ROOT)))

    print("\nNow score them:  python evaluation/run_evaluation.py --bootstrap 1000")
