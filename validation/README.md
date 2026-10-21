# Team validation labels (local only)

Everything in this folder except this file is gitignored: the labels are derived from the licensed
match video. They exist to choose parameters on more than the 17 public possessions
(EVALUATION.md); they never become the submission. RULES.md allows this: "Small amounts of
hand-labelled data used to train or validate a model are fine."

## Format (one CSV per labeller, e.g. `validation/alice.csv`)

    time,label
    25:00.0,H        from here Barcelona has the ball
    25:12.4,A
    25:20.0,D        ball out of play
    25:31.6,A        restart
    30:00.0,END      end of this segment

`time` is the tactical video's own clock (M:SS.s or H:MM:SS.s). Each segment ends with `END` and
must sit inside one half. Watch the tactical video locally; do not upload it anywhere.

## Rules (from CHALLENGE.md)

- H / A: that team has the ball and it is in play. D: the ball is out of play (stoppage).
- A possession changes only when a player *gains control*. A single touch, a deflection that is not
  intercepted, or a won aerial duel without control does not change it.
- Loose ball: the team that last had possession keeps it.
- The restarting team owns the ball from the restart (throw-in, corner, goal kick, free kick, kickoff).

## Workflow

1. Calibrate: label the public window (frames 11185..18684 = 7:27.4 to 12:27.4 of the video), then `python starter/make_validation_gt.py validation/<you>.csv -o validation/calib_gt.csv`
   and score it against the public GT. If the labeller's spellF1 is >= 0.8 the labels are trustworthy.
2. Label ~3 more 5-minute segments spread over the match (mid first half, start and end of second).
3. `python starter/make_validation_gt.py validation/*.csv` writes `validation/val_gt.csv`.
4. `python evaluation/run_evaluation.py --gt validation/val_gt.csv --out validation/scored --bootstrap 1000`
