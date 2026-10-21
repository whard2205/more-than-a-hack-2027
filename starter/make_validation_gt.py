"""Turn the team's hand-labelled change points into a ground truth the official scorer can read.

Why this exists: the public window is 17 possessions, and EVALUATION.md shows that tuning against it
picks the wrong smoothing window (smooth125 wins there, loses on the full match). Every parameter
choice in lever2/lever3 needs more possessions than that. RULES.md allows exactly this: "Small
amounts of hand-labelled data used to train or validate a model are fine and normal." These labels
choose parameters; they never become the submission.

Input: one change-point CSV per labeller, kept in validation/ (gitignored -- they are derived from
the licensed match video, so they stay local). See validation/README.md for the labelling rules.

    time,label
    25:00.0,H        Barcelona has the ball from here
    25:12.4,A
    25:20.0,D        ball out of play
    25:31.6,A        restart
    30:00.0,END      end of this segment -- frames after it are X until the next row

`time` is the tactical video's clock: H:MM:SS(.ff), M:SS(.ff), or plain seconds. Frame =
round(seconds * 25) (DATA.md: frame 0 of the tactical video is row 0 of a submission). A segment
runs from the first row after an END (or the top of the file) to the next END; every segment must
end with one, and must sit inside one graded half.

Output: frame,label, 199319 rows, X outside the labelled segments -- the same shape as
evaluation/gt/BAR-ATH_possession_gt_public.csv, so the official scorer reads it unchanged:

    python starter/make_validation_gt.py validation/alice.csv validation/bob.csv
    python evaluation/run_evaluation.py --gt validation/val_gt.csv --out validation/scored --bootstrap 1000
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from baseline import write_submission  # noqa: E402

LABELS = ("H", "A", "D")
END = "END"
DEFAULT_OUT = os.path.join(ROOT, "validation", "val_gt.csv")


def parse_time(text):
    """'1:23:45.2', '23:45.2' or '1425.2' -> seconds."""
    seconds = 0.0
    for part in text.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def read_segments(path, fps):
    """[(start_frame, end_frame_exclusive, [(frame, label), ...])] from one labeller's file."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or [f.strip() for f in reader.fieldnames] != ["time", "label"]:
            raise ValueError("{}: header must be exactly 'time,label'".format(path))
        rows = list(reader)

    segments, current, previous = [], [], None
    for line, row in enumerate(rows, start=2):
        label = (row["label"] or "").strip().upper()
        if label not in LABELS and label != END:
            raise ValueError("{}:{}: label {!r} is not H, A, D or END".format(
                path, line, row["label"]))
        frame = int(round(parse_time(row["time"]) * fps))
        if previous is not None and frame <= previous:
            raise ValueError("{}:{}: time {} is not after the row above".format(
                path, line, row["time"].strip()))
        previous = frame

        if label == END:
            if not current:
                raise ValueError("{}:{}: END with no open segment".format(path, line))
            segments.append((current[0][0], frame, current))
            current = []
        else:
            current.append((frame, label))
    if current:
        raise ValueError("{}: the last segment has no END row".format(path))
    return segments


def build(paths, game=GAME):
    """Full-length label array: X everywhere except inside the labelled segments."""
    fps = GAMES[game]["fps"]
    halves = [(first, last) for _, first, last in half_video_blocks(game)]
    labels = np.full(source_frames("tactical", game), "X", dtype="<U1")

    for path in paths:
        for start, end, points in read_segments(path, fps):
            if not any(first <= start and end - 1 <= last for first, last in halves):
                raise ValueError("{}: segment {}-{} is not inside one graded half {}".format(
                    path, start, end, halves))
            if np.any(labels[start:end] != "X"):
                raise ValueError("{}: segment {}-{} overlaps a segment from another file".format(
                    path, start, end))
            stops = [frame for frame, _ in points[1:]] + [end]
            for (frame, label), stop in zip(points, stops):
                labels[frame:stop] = label
    return labels


def possession_count(labels):
    """Maximal H or A runs -- the unit spellF1 scores. D and X both end a run."""
    alive = np.where(np.isin(labels, ("H", "A")), labels, "")
    starts = (alive != "") & (np.concatenate(([""], alive[:-1])) != alive)
    return int(np.sum(starts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+", help="one change-point CSV per labeller")
    parser.add_argument("-o", "--out", default=DEFAULT_OUT, help="where to write the ground truth")
    args = parser.parse_args()

    labels = build(args.labels)
    write_submission(args.out, labels)
    graded = labels != "X"
    print("wrote {}".format(os.path.relpath(args.out, ROOT)))
    print("  {} labelled frames ({:.1f} min): H {}  A {}  D {}".format(
        int(graded.sum()), graded.sum() / GAMES[GAME]["fps"] / 60,
        *(int(np.sum(labels == label)) for label in LABELS)))
    print("  {} possessions (the public window has 17)".format(possession_count(labels)))
