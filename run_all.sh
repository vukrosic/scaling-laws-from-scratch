#!/bin/sh
# Runs every step, the checks and the exercise, in the order of the video.
# Nothing here trains for long: the big table is read from files/runs.jsonl.
# To train the table again: python sweep.py cpu 4 8 16 32 64   (see README)
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_game.py step2_one_model.py step3_size.py step4_forecast.py \
         step5_text.py step6_formula.py step7_budget.py step8_schedule.py \
         step9_checks.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY $f
done
