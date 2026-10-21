"""Lever 3: find dead-ball stretches from the tracking itself, not only from Smart Tagging.

Lever 1 marks a stoppage only inside a tagged SET PIECES window, which covers 59 restarts. The public
window shows how much that leaves: F1 for the D class is 0.33 and six D transitions are missed, so
two possessions of the same team that a stoppage should separate come out merged (13 predicted vs 17
true). Two physical signals in the ATD say "the ball is dead" without any tag:

  * the ball leaves the pitch: its tracked x or y falls outside 0..1. On the public window that is
    true on 104 true-D frames and on 2 true-alive frames.
  * the players stop running: the mean speed over all tracked players drops well below what the
    team is running at around that moment. RELATIVE, not absolute: a first version with a fixed
    1.5 m/s cut-off looked excellent on the public window (macroF1 0.66 -> 0.84) but flagged 55% of
    the whole match as dead, because the players' median speed falls from ~2.0 m/s in the opening
    minutes to 1.0-1.7 m/s later and the public window is the fastest stretch of the match. The
    speed is now compared to its own rolling median over +-7.5 min.

Output is the same `DEAD` sentinel lever 1 writes, so lever 2's state machine consumes it unchanged.

    python starter/lever3_deadball.py

ponytail: one global speed threshold and one minimum duration for the whole match; a corner and a
substitution stop the players for very different lengths. Per-situation thresholds only if the
validation set (starter/make_validation_gt.py) shows the single pair is the bottleneck.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, require, source_frames  # noqa: E402
from lever1_tagging import DEAD  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, MISSING  # noqa: E402


def read_signals(game=GAME, data_dir=None):
    """(ball_x, ball_y, mean_speed), float32 arrays over video frames, NaN where unobserved.

    Same raw line format baseline.nearest_player_labels parses: "<frame>:<player>;...:<ball>", each
    group "x,y,speed" (speed in m/s, positions normalised 0..1). mean_speed averages over every
    tracked player on the frame, so duplicate or ghost tracks dilute it rather than break it.
    """
    raw_path = require(["atd_raw"], game, data_dir)[0]
    n_frames = source_frames("tactical", game)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    speed = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            ball = parts[2]
            if ball[0] != MISSING:
                fields = ball.split(",")
                ball_x[frame], ball_y[frame] = float(fields[0]), float(fields[1])
            speeds = []
            for group in parts[1].split(GROUP_SEP):
                if group and group[0] != MISSING:
                    value = group.split(",")[2]
                    if value[0] != MISSING:
                        speeds.append(float(value))
            if speeds:
                speed[frame] = sum(speeds) / len(speeds)
    return ball_x, ball_y, speed


# Measured on the public window (4 true stoppages, so a first guess, not a result): the rule is flat
# from ratio 0.65 to 0.80 -- precision ~0.8, recall 0.66-0.81, 3 of 4 stoppages found, 0 false-alarm
# runs -- and flags 24-35% of graded frames, in every 5-minute block between 14% and 58% instead of
# the fixed threshold's 20-79%. 0.70 is the middle of that plateau. Final values come from the team
# validation set, never from those 4 runs.
SPEED_WINDOW = 25         # frames averaged before comparing (1 s at 25 fps)
BASELINE_WINDOW = 15000   # rolling-median window for "what the players usually run at" (10 min)
SLOW_RATIO = 0.70         # slow = smoothed speed below this fraction of the local median
MIN_SLOW_FRAMES = 100     # a build-up is slow too; a stoppage stays slow for several seconds
BALL_OUT_PAD = 50         # frames on each side of the ball leaving the pitch (2 s)


def _runs(mask):
    """[(start, end_exclusive)] of every maximal True run."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def detect_dead(ball_x, ball_y, speed, slow_ratio=SLOW_RATIO, min_slow=MIN_SLOW_FRAMES,
                ball_pad=BALL_OUT_PAD, window=SPEED_WINDOW, baseline_window=BASELINE_WINDOW):
    """Boolean mask over frames: True where the tracking says the ball is dead."""
    filled = pd.Series(np.where(np.isnan(speed), np.nanmedian(speed), speed))
    smoothed = filled.rolling(window, center=True, min_periods=1).mean().to_numpy()
    usual = pd.Series(smoothed).rolling(baseline_window, center=True, min_periods=1).median()

    slow = np.zeros(len(speed), dtype=bool)
    for start, end in _runs(smoothed < slow_ratio * usual.to_numpy()):
        if end - start >= min_slow:
            slow[start:end] = True

    out = (~np.isnan(ball_x)) & ((ball_x < 0) | (ball_x > 1) | (ball_y < 0) | (ball_y > 1))
    near_out = np.convolve(out.astype(np.int32), np.ones(2 * ball_pad + 1, dtype=np.int32),
                           mode="same") > 0
    return slow | near_out


def apply_deadball(codes, mask):
    """codes with DEAD written wherever the mask is True (lever 1's DEAD frames are kept)."""
    out = codes.copy()
    out[mask] = DEAD
    return out


if __name__ == '__main__':
    from baseline import build, write_submission, RESULTS_DIR, LABELS
    from lever1_tagging import apply_smart_tagging
    from lever2_sticky import sticky_labels, THRESHOLDS

    codes, _ = apply_smart_tagging(build())
    mask = detect_dead(*read_signals())
    codes = apply_deadball(codes, mask)
    print("dead-ball detector: {} frames flagged by the tracking signals, {} DEAD in total\n".format(
        int(mask.sum()), int(np.sum(codes == DEAD))))

    for threshold in THRESHOLDS:
        labels = sticky_labels(codes, threshold)
        name = "dead_h{}".format(threshold)
        path = write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        counts = {label: int(np.sum(labels == label)) for label in LABELS}
        print("{:<16} threshold {:<4} H {:>6}  A {:>6}  D {:>6}".format(
            name, threshold, counts["H"], counts["A"], counts["D"]))
        print("{:<16} -> {}".format("", os.path.relpath(path, ROOT)))

    print("\nNow score them:  python evaluation/run_evaluation.py --bootstrap 1000")
