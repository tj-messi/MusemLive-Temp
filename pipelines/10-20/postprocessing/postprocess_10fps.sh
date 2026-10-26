#!/usr/bin/env bash
set -Eeuo pipefail

: "${MUSEUMLIVE_10FPS_RUN_ROOT:?set the external 10 FPS run root}"
: "${OPENROUTER_API_KEY:?set the judge API key in the process environment}"
: "${MUSEUMLIVE_TASKS_FILE:?set the frozen final450 task file}"
: "${MUSEUMLIVE_WORK_ROOT:?set the external MuseumLive work root}"

pipeline_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=$MUSEUMLIVE_10FPS_RUN_ROOT
judge_py="$pipeline_dir/evaluators/judge_points.py"
analysis_dir=${MUSEUMLIVE_ANALYSIS_DIR:-$pipeline_dir/postprocessing}
runner_dir=${MUSEUMLIVE_ANALYSIS_RUNNER_DIR:-$pipeline_dir/runners}
python_bin=${MUSEUMLIVE_PYTHON:-python3}
log="$run_root/postprocess_10fps.log"
rubric_file=${MUSEUMLIVE_RUBRIC_FILE:-}
if [ -z "$rubric_file" ]; then
  if [ -s "$run_root/runs/main_minicpm/judge_points_lenient_v3/rubric.json" ]; then
    rubric_file="$run_root/runs/main_minicpm/judge_points_lenient_v3/rubric.json"
  else
    # A lenient result may be a deterministic rescore and therefore share the
    # original Judge rubric without carrying a duplicate rubric.json.
    rubric_file="$run_root/runs/main_minicpm/judge_points/rubric.json"
  fi
fi

[ -s "$MUSEUMLIVE_TASKS_FILE" ] || {
  printf 'frozen task file not found or empty: %s\n' "$MUSEUMLIVE_TASKS_FILE" >&2
  exit 1
}
[ -s "$rubric_file" ] || {
  printf 'frozen rubric file not found or empty: %s\n' "$rubric_file" >&2
  exit 1
}
export MUSEUMLIVE_RUBRIC_FILE=$rubric_file

note() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$log"
}

required_runs=(
  main_minicpm main_venus main_gander main_moss
  force_venus force_gander force_minicpm
  mute_minicpm black_minicpm notext_moss black_moss
  blind_minicpm blind_moss
)
declare -A expected_slots
for run in main_minicpm main_venus main_gander main_moss \
  force_venus force_gander force_minicpm mute_minicpm black_minicpm \
  notext_moss black_moss; do
  expected_slots[$run]=1250
done
expected_slots[blind_minicpm]=1000
expected_slots[blind_moss]=1000

while true; do
  ready=0
  for run in "${required_runs[@]}"; do
    summary="$run_root/runs/$run/judge_points_lenient_v3/summary.json"
    if [ -s "$summary" ] && "$python_bin" - "$summary" "${expected_slots[$run]}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
complete = (
    summary.get("slots") == expected
    and summary.get("scorable_ok") == expected
    and summary.get("api_failed") == 0
)
raise SystemExit(0 if complete else 1)
PY
    then
      ready=$((ready + 1))
    fi
  done
  note "waiting for scheduled 10 FPS eval+judge: $ready/${#required_runs[@]} complete"
  [ "$ready" -eq "${#required_runs[@]}" ] && break
  sleep 60
done

note "starting alternate judges for four 10 FPS main-model outputs"
judge_pids=()
for judge in tencent/hy3 deepseek/deepseek-v4-pro z-ai/glm-5.2 moonshotai/kimi-k2.6; do
  judge_name=${judge//\//_}
  (
    for model in minicpm venus gander moss; do
      out="$run_root/judges/$judge_name/$model"
      mkdir -p "$out"
      JUDGE_EXTRA='{"reasoning":{"enabled":false}}' JUDGE_MODEL="$judge" \
        "$python_bin" -u "$judge_py" run \
          --eval-glob "$run_root/runs/main_$model/shard*/eval/evaluation_results.json" \
          --out "$out" --workers 12 --profile lenient_v3 \
          > "$out/judge.log" 2>&1
    done
  ) &
  judge_pids+=("$!")
done

note "building and starting 10 FPS recent-30-second reruns"
EXPERIMENT_OUTPUT_ROOT="$run_root" "$python_bin" "$runner_dir/recent30_tasks.py" make
EXPERIMENT_OUTPUT_ROOT="$run_root" MAIN25_ROOT="$run_root/runs" VISUAL_FPS=10 \
  EVAL_GPUS="${EVAL_GPUS:-2,3,4,5,6,7}" \
  "$python_bin" -u "$runner_dir/run_recent30.py" \
  >> "$run_root/recent30_10fps.log" 2>&1

note "starting offline upper-bound regeneration from 10 FPS timelines"
EXPERIMENT_OUTPUT_ROOT="$run_root" MAIN25_ROOT="$run_root/runs" VISUAL_FPS=10 \
  "$python_bin" -u "$analysis_dir/offline_upper.py" --workers 16 \
  >> "$run_root/offline_upper_10fps.log" 2>&1

mkdir -p "$run_root/runs/offline_qwen_omni/judge_points_lenient_v3"
"$python_bin" -u "$judge_py" run \
  --eval-glob "$run_root/runs/offline_qwen_omni/eval/evaluation_results.json" \
  --out "$run_root/runs/offline_qwen_omni/judge_points_lenient_v3" \
  --workers 16 --profile lenient_v3 \
  >> "$run_root/runs/offline_qwen_omni/judge_points_lenient_v3/judge.log" 2>&1

for pid in "${judge_pids[@]}"; do
  wait "$pid"
done
note "alternate judges, recent30, and offline upper bound complete"

EXPERIMENT_OUTPUT_ROOT="$run_root" MAIN25_ROOT="$run_root/runs" JUDGE_DIR_NAME=judge_points_lenient_v3 \
  "$python_bin" "$analysis_dir/analysis_a.py" > "$run_root/analysis_a.log" 2>&1
EXPERIMENT_OUTPUT_ROOT="$run_root" MAIN25_ROOT="$run_root/runs" JUDGE_DIR_NAME=judge_points_lenient_v3 \
  "$python_bin" "$analysis_dir/analysis_b.py" > "$run_root/analysis_b.log" 2>&1
EXPERIMENT_OUTPUT_ROOT="$run_root" JUDGE_DIR_NAME=judge_points_lenient_v3 \
  "$python_bin" "$analysis_dir/build_report_1_7.py" > "$run_root/build_report.log" 2>&1
EXPERIMENT_OUTPUT_ROOT="$run_root" JUDGE_DIR_NAME=judge_points_lenient_v3 \
  "$python_bin" "$analysis_dir/verify_1_7.py" > "$run_root/verify_1_7.log" 2>&1

note "all 10 FPS Experiment 1–7 analyses, reports, and validation complete"
