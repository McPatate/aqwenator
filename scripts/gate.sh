#!/usr/bin/env bash
# Correctness gate: aq-check on every reference file, in the default configuration and under
# the scheduling and speculation variants, plus one fault-injection leg that must fail.
# Usage: scripts/gate.sh <aq-check binary> <dir with reference .safetensors from golden.py>
#   Reference files are named <model>[-variant].safetensors with the model id's "/" and the
#   organization and dots dropped, lowercased, e.g. qwen35-9b-long.safetensors for Qwen/Qwen3.5-9B;
#   set MODEL_ORG (default Qwen) for another organization.
set -uo pipefail
CHECK=${1:?aq-check binary}
GOLDEN=${2:?reference directory}
ORG=${MODEL_ORG:-Qwen}
export AQ_GATE=1
pass=0 fail=0
run() { # <env string> <reference file> <expect: pass|fail>
  local out
  out=$(env $1 timeout "${GATE_TIMEOUT:-1500}" "$CHECK" "$3" "$2" 2>&1)
  local code=$?
  echo "== $(basename "$2") [$1] exit=$code"
  echo "$out" | grep -E "teacher|greedy vs|batched seq|session|shared|pressure|specul|stop under|PASS|FAIL|MISMATCH|rror|panic" | cut -c1-230
  if { [ "$4" = pass ] && [ $code -eq 0 ]; } || { [ "$4" = fail ] && [ $code -ne 0 ]; }; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "   UNEXPECTED"; fi
}
last=
for env in "AQ_GATE=1" "AQ_SRPT=1 AQ_BURST_PREFILL=2048 AQ_BURST_EVERY=2" "AQ_CHECK_BUDGET=1000 AQ_CHECK_CHUNK=777" "AQ_MTP=0" "AQ_MTP_K=1" "AQ_KV=int8"; do
  for g in "$GOLDEN"/*.safetensors; do
    case $(basename "$g") in *trace*) continue ;; esac
    # qwen35-9b-long.safetensors -> Qwen/Qwen3.5-9B, looked up among the cached models
    stem=$(basename "$g" .safetensors | sed -E 's/-(long|16k)$//; s/\.//g')
    model=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--"$ORG"--* 2>/dev/null | sed 's/.*models--//; s/--/\//' | awk -v s="$stem" '{ m = tolower($0); sub(/.*\//, "", m); gsub(/\./, "", m); if (m == s) print }' | head -1)
    [ -n "$model" ] || { echo "no cached model for $g"; fail=$((fail + 1)); continue; }
    run "$env" "$g" "$model" pass
    last="$g $model"
  done
done
# The check must notice a corrupted speculative token.
[ -n "$last" ] && run "AQ_SPEC_FAULT=40" ${last% *} "${last#* }" fail
echo "gate: $pass as expected, $fail unexpected"
[ $fail -eq 0 ]
