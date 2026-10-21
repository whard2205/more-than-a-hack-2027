# Remontada Labs — Initial Challenge technical document

Team: Remontada Labs. Challenge: label every frame of BAR-ATH as `H`/`A`/`D`/`X` (More than a Hack
2027, Phase 1).

## Approach

Four stages, each a separate script in `starter/`, each independently runnable and scored:

1. **`baseline.py`** (provided) — nearest tracked player to the ball, per frame. Where the ball is
   not visible (43% of graded frames), no answer (`NOBODY`).
2. **`lever1_tagging.py`** — reads the Smart Tagging XML's `SET PIECES` instances. Any `NOBODY` run
   that falls inside a tagged set-piece window is the stoppage itself: marked `DEAD`, a sentinel
   that is always written as `D`, regardless of what a smoothing pass around it would otherwise
   vote. We do **not** use the Team tag or paint the whole tagged interval: every non-kickoff
   `SET PIECES` instance in this file is a fixed 875-frame (35 s) clip, not the real dead-ball
   span — measured directly, not assumed — so trusting its boundaries for team ownership would
   overwrite real, possibly contested, play with a guess. Only touching frames the baseline was
   already blind on means this step can only add information, never remove a confident read.
3. **`lever2_sticky.py`** — replaces baseline's symmetric majority-vote smoothing with a causal
   hysteresis filter. Possession only changes once the *other* side has had `threshold` consecutive
   frames of nearest-player evidence; a short flicker (a deflected pass, a contested header) can
   never flip it. `DEAD` frames (from step 2) reset the state to "no current possessor" — the next
   possession is only assigned once real evidence appears after the restart, never inherited from
   who had it before the stoppage. `NOBODY` frames hold the current state without reinforcing or
   challenging it. **Back-dated:** a flip only clears after `threshold` frames, but the possession
   it detects began at the first of them, so the streak is relabelled back to where it started
   (we label the whole match offline, so rewriting is free). Without this every H↔A change landed
   `threshold` frames late — 31–70 frames on the public window, past the scorer's 25-frame
   tolerance, so each counted as one missed *and* one spurious transition.
4. **`lever3_deadball.py`** *(candidate, not yet the submission)* — finds stoppages from the
   tracking itself, outside the 59 tagged set pieces: the ball leaving the pitch (±2 s) and the
   players running well below their own local pace (mean speed under 0.70 × its rolling median
   over ±7.5 min, for at least 4 s). Writes the same `DEAD` sentinel lever 1 does.

## Definition of possession / control

Per `CHALLENGE.md`: a possession is a maximal run of one team with the ball alive; a single touch
without control, a deflected pass that isn't intercepted, or a contested aerial duel does not
change it. We operationalise "control" as *sustained* nearest-player evidence — `threshold`
straight frames (25 fps) favouring the other side — rather than any single frame's geometric
nearest neighbour. This is the entire justification for lever 2: a one-off closest-player reading
during a 50/50 duel is exactly the kind of touch the challenge definition says must not flip
possession, and a hysteresis filter is the direct encoding of that rule, rather than an incidental
side effect of a wide smoothing window (which also blurs every real transition by the same amount).

## Assets used / dropped

| Asset | Used | Why |
|---|---|---|
| Tactical ATD (tracking) | Yes | Core signal: nearest player to the ball |
| Tactical ATD metadata (XML) | Yes | Team-of-track-slot, team names |
| Smart Tagging XML | Yes, partially | `SET PIECES` instance boundaries only, to confirm real stoppages. Not its `Team` tag, not `GOALS`, not any other code (`BUILD UP`, `PROGRESSION`, …) — dropped, see below |
| Homography | No | Not needed: distances are compared in the tracking's own normalised pitch coordinates (scaled to metres), not pixel space |
| Tactical video | No | Not needed by this pipeline |
| Broadcast recording (video/ATD/homography) | No | A different, unsynchronised timeline (164,488 frames vs. 199,319); mixing it in without a frame mapping would misalign labels |
| LLMs | No | Not used anywhere in the labelling pipeline |

**Why the rest of Smart Tagging is dropped for Phase 1:** the non-set-piece codes (`BUILD UP`,
`PROGRESSION`, `ATTACKING TRANSITION`, …) describe phases of play, not possession, and per
`load_smart_tagging.py`'s own documentation are not ground truth and can span, fall short of, or
cover more than one real possession. Using them as a possession signal without a validated mapping
risked adding noise rather than removing it, for no measured gain we could point to.

## Hyperparameter selection (and what we did *not* do)

`EVALUATION.md` and the public ground truth's own 17 possessions are explicit that small
differences there are noise. We treated the public window as a **sanity check and a coarse
direction-finder**, never as a tuning set to chase a specific number:

- Lever 2's threshold was swept from 10 to 150 frames (re-swept after back-dating). Score rises
  smoothly to a **flat plateau at 50–65 frames** (poss=13, duration 22.2 s, spellF1 0.520 at every
  point in that range, 0.507 at 75), then falls off. A multi-point plateau, not one lucky value,
  is why we trust this as a real effect of the mechanism rather than a fit to 17 possessions.
  Shipped threshold: **50 frames (2.0 s)**, the start of the plateau.
- We did not search sub-frame thresholds, per-class thresholds (home vs. away), or per-restart-type
  thresholds (a goal kick's first phase plausibly needs a different value than a throw-in's) —
  flagged as the next place to spend effort if the private full-match score disagrees with this
  window's ranking.

## Results on the public 5-minute ground truth

| Submission | macroF1 | spellF1 | possessions (gt 17) | mean duration (gt 14.2 s) |
|---|---|---|---|---|
| baseline_raw (provided) | 0.597 | 0.038 | 131 | 2.0 s |
| baseline_smooth125 (provided, best of 4) | 0.581 | 0.424 | 16 | 18.5 s |
| lever1 (tagging_smooth125) | 0.651 | 0.449 | 16 | 18.0 s |
| **lever1 + lever2 back-dated (sticky_h50)** | **0.657** | **0.520** | **13** | **22.2 s** |
| lever1 + lever2 + lever3 (dead_h50) — *candidate* | 0.824 | 0.516 | 14 | 17.7 s |

The headline gain (spellF1 0.467 vs. 0.424) is below this window's own bootstrap noise band — 17
possessions cannot confirm it, which is exactly what `EVALUATION.md` warns about. What we trust
more: **macroF1 and the per-class D-score rise consistently across every smoothed/sticky variant we
tried and never regress relative to their baseline counterpart** (`macroF1` is computed over 7,500
graded frames on this window, not 17 possessions, so it is far less sensitive to this sample being
small). Final ranking is decided by the private full-match score, per the rules.

**Lever 3 is not adopted yet, and why.** On this window it lifts the D-class F1 from 0.33 to 0.76 and
macroF1 from 0.657 to 0.824 while leaving spellF1 flat (0.520 → 0.516, inside the noise). But its
parameters were chosen from the window's four true stoppages, so that gain is not evidence. A first
version with a *fixed* 1.5 m/s speed cut-off produced an even better-looking window score
(macroF1 0.84) while flagging 55% of the whole match as dead: players' median speed falls from ~2.0
m/s in the opening minutes to 1.0–1.7 m/s later, and the public window is the fastest stretch. We
caught it only by checking the flagged share in every 5-minute block, and replaced it with the
relative threshold above (flagged share 14–58% per block instead of 20–79%; regression test
`test_gradual_slowdown_is_not_a_stoppage`). It is adopted only if it also improves on the team
validation set (`starter/make_validation_gt.py`, hand labels used for validation only, kept local).

## Limitations

- **Ball invisible on 43% of graded frames.** Both levers treat an invisible ball as "hold current
  state", which is right during a genuine loose-ball/crowded scramble but wrong if the ball is
  simply off-screen while play continues normally elsewhere — we have no signal to distinguish the
  two cases from tracking data alone. We tested whether the long (>20s) NOBODY runs that fall
  *outside* every tagged `SET PIECES` window were missed stoppages worth marking `DEAD` too (Smart
  Tagging's own documentation admits false negatives). They are not: the three longest ones sit
  right next to `BALL IN THE BOX` / `PLAYERS IN THE BOX` tags — goalmouth scrambles, i.e. the ball
  is in its most active moments, just occluded by bunched players. Marking them `DEAD` would have
  been a confident wrong guess exactly where lever 2's "hold the last possessor" default is the
  safer one. Left unchanged; recorded here so nobody re-tries this and ships it without checking.
- **No real player identity.** ATD metadata slots are tracklets ("Track 1".."Track 46"), not
  persistent player IDs, so no model here uses anything about *which* player, only which team.
- **Smart Tagging is not ground truth** (false positives/negatives acknowledged by its own
  documentation); we only trust its interval *boundaries* for stoppages, never its Team tag.
- **Lever 2's threshold is a single flat constant**, not adapted to set-piece type or match phase.
- **Both halves of the pipeline are evaluated only on the public 5-minute window** before
  submission; the real ranking is on the private full-match ground truth.

## Reproduction

```bash
git clone <this repository>
cd more-than-a-hack-2027
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# unpack the competition data into ./data (see DATA.md) or export MTAH_DATA_DIR
.venv/bin/python starter/baseline.py           # writes the 4 provided baselines
.venv/bin/python starter/lever1_tagging.py     # writes tagging_{raw,smooth25,smooth50,smooth125}
.venv/bin/python starter/lever2_sticky.py      # writes sticky_{h35,h50,h75} -- h50 is the submission
.venv/bin/python starter/lever3_deadball.py    # writes dead_{h35,h50,h75} -- candidate, see Results
.venv/bin/python evaluation/run_evaluation.py --bootstrap 1000
.venv/bin/python evaluation/validate_submission.py evaluation/results/sticky_h50.csv
```

Self-checks (no data download required, synthetic + smoke tests):

```bash
.venv/bin/python starter/test_lever2_sticky.py    # synthetic, runs anywhere
.venv/bin/python starter/test_lever3_deadball.py  # synthetic, runs anywhere
.venv/bin/python starter/test_make_validation_gt.py  # needs only the public ground truth
.venv/bin/python starter/test_lever1_tagging.py   # needs the real data on disk
```

Submitted file: `evaluation/results/sticky_h50.csv`, renamed to `<team_name>.csv` per `SUBMISSION.md`.
