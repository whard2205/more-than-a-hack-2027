"""Smallest check that would fail if make_validation_gt.py's conversion broke.

The strongest test available: turn the public ground truth into change points, convert them back,
and require the result to be identical to the public ground truth, frame for frame. No match data
needed -- the public ground truth ships with the repository.

    python starter/test_make_validation_gt.py
"""

import csv
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_validation_gt import build, possession_count  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def _write(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    handle.write("time,label\n")
    handle.writelines("{},{}\n".format(t, label) for t, label in rows)
    handle.close()
    return handle.name


def _raises(rows, fragment):
    path = _write(rows)
    try:
        build([path])
    except ValueError as error:
        assert fragment in str(error), "wrong error: {}".format(error)
        return
    finally:
        os.unlink(path)
    raise AssertionError("expected an error containing {!r}".format(fragment))


def test_public_ground_truth_round_trips_exactly():
    gt = np.array([row["label"] for row in csv.DictReader(open(PUBLIC_GT))])
    graded = np.where(gt != "X")[0]
    first, last = graded[0], graded[-1]
    rows = [("{:.2f}".format(f / 25), gt[f]) for f in range(first, last + 1)
            if f == first or gt[f] != gt[f - 1]]
    rows.append(("{:.2f}".format((last + 1) / 25), "END"))

    path = _write(rows)
    try:
        labels = build([path])
    finally:
        os.unlink(path)
    assert np.array_equal(labels, gt), "round trip differs on {} frames".format(
        int(np.sum(labels != gt)))
    assert possession_count(labels) == 17  # DATA.md: the public window has 17 possessions


def test_clock_formats_agree():
    # 25:00.0, 0:25:00.0 and 1500 are the same instant: frame 37500.
    for text in ("25:00.0", "0:25:00.0", "1500"):
        path = _write([(text, "H"), ("25:04.0", "END")])
        try:
            labels = build([path])
        finally:
            os.unlink(path)
        assert labels[37499] == "X" and labels[37500] == "H" and labels[37599] == "H"
        assert labels[37600] == "X"


def test_rejects_bad_input():
    _raises([("25:00", "H"), ("25:10", "B"), ("25:20", "END")], "is not H, A, D or END")
    _raises([("25:00", "H"), ("24:59", "A"), ("25:20", "END")], "is not after")
    _raises([("25:00", "H"), ("25:10", "A")], "no END")
    _raises([("25:00", "END")], "no open segment")
    # 60:00 -> 89:59 crosses halftime (first half ends at frame 90651, about 60:26)
    _raises([("60:00", "H"), ("90:00", "END")], "not inside one graded half")


def test_rejects_overlap_between_labellers():
    a = _write([("25:00", "H"), ("26:00", "END")])
    b = _write([("25:30", "A"), ("27:00", "END")])
    try:
        build([a, b])
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("overlap was not caught")
    finally:
        os.unlink(a)
        os.unlink(b)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  " + test.__name__)
    print("\n{} checks passed".format(len(tests)))
