#!/usr/bin/env bash
# Whole pipeline for one phone clip: hand tracking -> demos -> replay on several arms ->
# sim-verified data multiplication -> BC policy -> evaluation -> self-improvement.
#
#   scripts/run_all.sh data/raw/stack_main.mp4 # real clip (data/raw/stack_main_events.json overrides segmentation)
#   scripts/run_all.sh --synthetic [N_REPS]    # same pipeline on N_REPS synthetic reps (default 10)
#
# Sizes (N_DATAGEN, N_EVAL, ROUNDS, ROLLOUTS, ROBOTS, UPDATES) come from the environment, e.g. N_EVAL=50 ROUNDS=1 scripts/run_all.sh --synthetic
# Outputs: data/processed/<clip>_*, outputs/<clip>/{replay,eval}/ and outputs/perception/<clip>_*.
set -euo pipefail

N_DATAGEN="${N_DATAGEN:-1600}"
N_EVAL="${N_EVAL:-50}"
ROUNDS="${ROUNDS:-1}"
ROLLOUTS="${ROLLOUTS:-100}"
ROBOTS="${ROBOTS:-Panda UR5e IIWA Kinova3}"
UPDATES="${UPDATES:-4000}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ $# -lt 1 ]]; then
    sed -n '2,9p' "$0"
    exit 1
fi

if [[ "$1" == "--synthetic" ]]; then
    reps="${2:-10}"
    clip="synthetic${reps}"
    extract_args=(--synthetic "$reps")
    video_args=()
else
    video="$1"
    clip="$(basename "${video%.*}")"
    extract_args=(--video "$video")
    video_args=(--video "$video" --track "data/processed/${clip}_track.npz")
fi

demos="data/processed/${clip}_demos.npz"
episodes="data/processed/${clip}_episodes.npz"
policy="data/processed/${clip}_bc.pt"
out="outputs/${clip}"

stage() { echo; echo "=== $1  ($(date +%H:%M:%S))"; }

stage "1/6 phone video -> table-frame demos"
python scripts/01_extract_demos.py "${extract_args[@]}"

stage "2/6 open-loop replay of the retargeted demos on: $ROBOTS"
python scripts/02_replay_demos.py --demos "$demos" "${video_args[@]}" --robots $ROBOTS \
    --out "$out/replay"

stage "3/6 data multiplication over $N_DATAGEN random layouts"
python scripts/03_generate_data.py --demos "$demos" --n "$N_DATAGEN" --out "$episodes"

stage "4/6 behaviour cloning"
python scripts/04_train_bc.py --episodes "$episodes" --out "$policy" --updates "$UPDATES"

stage "5/6 evaluation on $N_EVAL held-out layouts (replay vs BC, pushes, data scaling)"
python scripts/05_evaluate.py --policy "$policy" --demos "$demos" --episodes "$episodes" \
    --n-eval "$N_EVAL" --scaling --updates "$UPDATES" --out "$out/eval/results.json"

stage "6/6 success-filtered self-improvement, $ROUNDS round(s)"
python scripts/06_self_improve.py --episodes "$episodes" --policy "$policy" --rounds "$ROUNDS" \
    --rollouts "$ROLLOUTS" --updates "$UPDATES" --n-eval "$N_EVAL" --work-dir "data/processed/${clip}_self_improve" \
    --out "$out/eval/self_improve.json"

stage "done: results in $out/"
