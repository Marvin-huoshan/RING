#!/usr/bin/env bash
# set your fixed params here:
FRAC=0.2
NATT=4
EPS=5.0
CLIP=10.0
LR=0.01

# loop over your three directories:
for d in dir prob qty; do
  cd "$d" || continue
  for f in *_analysis_*_[0-9].txt; do
    # e.g. f="Geminiguard_analysis_Input_3.txt"
    def="${f%%_analysis*}"
    rest="${f#*_analysis_}"        # "Input_3.txt"
    atk="${rest%_*}"               # "Input"
    rnd="${rest##*_}"              # "3.txt"
    rnd="${rnd%.txt}"              # "3"
    new="${def}_analysis_${atk}_frac=${FRAC}_nattacker=${NATT}_epsilon=${EPS}_clip=${CLIP}_lr=${LR}_round_${rnd}.txt"
    mv -- "$f" "$new"
  done
  cd - >/dev/null
done
